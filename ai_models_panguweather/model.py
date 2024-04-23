# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import onnxruntime as ort
from ai_models.model import Model
import xarray as xr

LOG = logging.getLogger(__name__)


class PanguWeather(Model):
    # Download
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/pangu-weather/{file}"
    )
    download_files = ["pangu_weather_24.onnx", "pangu_weather_6.onnx"]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ["msl", "10u", "10v", "2t"] #Checked with github page of PanguWeather
    param_level_pl = (
        ["z", "q", "t", "u", "v"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    # Output
    expver = "pguw"

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        
        # Adding the differents steps possible for PanguWeather
        if not("submodels" in kwargs):
            self.steps = [1,3,6,24]
        else:
            if not(kwargs["submodels"].isinstance(list)):
                raise ValueError("submodels must be a list, containing lead_times of each model")
            else:
                self.steps = list(map(int, kwargs["submodels"]))

    def run(self):
        fields_pl = self.fields_pl

        param, level = self.param_level_pl
        
        if isinstance(self.all_fields, list):
            self.param_sfc = ["msl", "u10", "v10", "t2m"]
            fields_pl = fields_pl.sel(isobaricInhPa=level)[param]
            fields_sfc = self.fields_sfc[self.param_sfc]
            
            fields_pl_numpy = np.concatenate([fields_pl[f].values for f in param])
            fields_sfc_numpy = np.concatenate([fields_sfc[f].values for f in self.param_sfc])
            
        else:
            fields_pl = fields_pl.sel(param=param, level=level)
            fields_pl = fields_pl.order_by(param=param, level=level)

            fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
            fields_pl_numpy = fields_pl_numpy.reshape((5, 13, 721, 1440))

            fields_sfc = self.fields_sfc
            fields_sfc = fields_sfc.sel(param=self.param_sfc)
            fields_sfc = fields_sfc.order_by(param=self.param_sfc)

            fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)

        input = fields_pl_numpy
        input_surface = fields_sfc_numpy

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.num_threads
           
        if self.post_processing:
            if not isinstance(self.all_fields, list):
                raise ValueError("Currently, post-processing mode is only available for file input.")
            
            data_vars = {}
            # Switch to post-processing mode, and downloading the different models
            pangu_weathers = {lt:os.path.join(self.assets, f"pangu_weather_{lt}.onnx") for lt in self.steps}
            for lt in pangu_weathers:
                os.stat(pangu_weathers[lt])
            
            with self.timer(f"Loading PanguWeather models for lead times {self.steps}..."):
                if self.test_mode:
                    print("\n *** TEST MODE: not loading models because test mode on. ***\n")
                else:
                    ort_sessions = {lt:ort.InferenceSession(
                        pangu_weathers[lt],
                        sess_options=options,
                        providers=self.providers,
                    ) for lt in pangu_weathers}
            
            data_vars = {}
            surface_outputs = []
            pl_outputs = []
            
            lead_times = self.lead_times
            calculated = {0: (input, input_surface)}
            count = 0
            
            with self.ppstepper(self.steps) as ppstepper:
                if self.test_mode:
                    print("\n *** TEST MODE: ppstepper.stepping is: " + str(ppstepper.stepping) + " ***\n")
                    print("\n *** TEST MODE: calculated is: " + str(calculated.keys()) + " ***\n")
                for lt in lead_times:
                    stepping = ppstepper.stepping[lt] #ppstepper knows lt
                    # as it is intialised with PanguWeather.lead_times
                    # /!\ here, stepping refers to the stepping for lt
                    max_calc_i = 0
                    if self.test_mode:
                        print("\n *** TEST MODE: stepping is: " + str(stepping) + " , lead_time is " + str(lt) + " ***\n")
                    
                    for i in range(len(stepping) - 1, -1, -1):
                        # We are looking for the nearest calculated step
                        if stepping[i] in calculated.keys():
                            max_calc_i = i
                            if self.test_mode:
                                print("\n *** TEST MODE: Nearest calculated step: " + str(max_calc_i) + " ***\n")
                            break
                    
                    for i in range(max_calc_i + 1, len(stepping)):
                        count += 1
                        if self.test_mode:
                            print("\n *** TEST MODE: Value of i: " + str(i) + ", value of stepping[i]: " + str(stepping[i]) + " *** \n")
                        inputs = calculated[stepping[i-1]]
                        if self.test_mode:
                            output, output_surface = "test" + str(stepping[i-1]), "test" + str(stepping[i-1])
                        else:
                            output, output_surface = ort_sessions[stepping[i] - stepping[i-1]].run(
                                None,
                                {
                                    "input": inputs[0],
                                    "input_surface": inputs[1],
                                },
                            )
                        calculated[stepping[i]] = (output, output_surface)

            if self.test_mode:
                print("\n ****************** Finished post-processing in test mode. ******************\n")
                return
            # Adapted from Louis fork.
            surface_output = np.stack([calculated[lt][1] for lt in lead_times]).transpose((1, 0, 2, 3))
            pl_output = np.stack([calculated[lt][0] for lt in lead_times]).transpose((1, 0, 2, 3, 4))
            for i in range(len(self.param_sfc)):
                data_vars[self.param_sfc[i]] = (
                    ("time", "lat", "lon"),
                    surface_output[i],
                )
            for i in range(len(self.param_level_pl[0])):
                data_vars[self.param_level_pl[0][i]] = (
                    ("time", "level", "lat", "lon"),
                    pl_output[i],
                )   
            times = [self.all_fields[1].time.values[0] + np.timedelta64(lead_times[i], 'h') for i in range(len(lead_times))]
            
            lat, lon = self.all_fields[0].latitude.values[::-1], self.all_fields[0].longitude.values
            saved_xarray = xr.Dataset(
                data_vars=data_vars,
                coords=dict(
                    lon=lon,
                    lat=lat,
                    time=times,
                    level=self.param_level_pl[1],
                ),
            )

            saved_xarray = saved_xarray.reindex(level=saved_xarray.level[::-1])
            saved_xarray = saved_xarray.rename({"level": "isobaricInhPa"})
            start_date = self.all_fields[0].valid_time.values[0] # May be a problem if working with monthly files
            
            # ONNX cannot handle the four models PW1, PW3, PW6, PW24, so the forecasts are splitted in two files
            if max(lead_times) < 24:
                name = os.path.join(self.path, str(self.date)[:4], str(self.date)[4:6], f"pangu_weather_ST_{np.datetime64(start_date, 'h')}.nc")
            else:
                name = os.path.join(self.path, str(self.date)[:4], str(self.date)[4:6], f"pangu_weather_LT_{np.datetime64(start_date, 'h')}.nc")
            os.makedirs(os.path.dirname(name), exist_ok=True)
                
            LOG.info(f"Saving to {name}")
            encoding = {}
            encoding = {}
            # for data_var in data_vars: # Louis compression, not used here
            #     encoding[data_var] = {
            #     "original_shape": saved_xarray[data_var].shape,
            #     "_FillValue": -32767,
            #     "dtype": np.float16,
            #     "add_offset": saved_xarray[data_var].mean().compute().values,
            #     "scale_factor": saved_xarray[data_var].std().compute().values / 1000, # save up to 32 std
            #     # "zlib": True,
            #     # "complevel": 5,
            #     }
            # saved_xarray.to_netcdf(name, engine="netcdf4", mode="w", encoding=encoding, compute=True)
            saved_xarray.to_netcdf(name, engine="netcdf4", mode="w", compute=True)

            return

        else:
            # Louis adaption from here, not used in post-processing mode
            
            pangu_weather_24 = os.path.join(self.assets, "pangu_weather_24.onnx")
            pangu_weather_6 = os.path.join(self.assets, "pangu_weather_6.onnx")

            # That will trigger a FileNotFoundError

            os.stat(pangu_weather_24)
            os.stat(pangu_weather_6)

            with self.timer(f"Loading {pangu_weather_24}"):
                ort_session_24 = ort.InferenceSession(
                    pangu_weather_24,
                    sess_options=options,
                    providers=self.providers,
                )

            with self.timer(f"Loading {pangu_weather_6}"):
                ort_session_6 = ort.InferenceSession(
                    pangu_weather_6,
                    sess_options=options,
                    providers=self.providers,
                )

            input_24, input_surface_24 = input, input_surface
            
            if isinstance(self.all_fields, list):
                data_vars = {}
                surface_outputs = []
                pl_outputs = []

            with self.stepper(6) as stepper:
                for i in range(self.lead_time // 6):
                    step = (i + 1) * 6

                    if (i + 1) % 4 == 0: 
                        output, output_surface = ort_session_24.run(
                            None,
                            {
                                "input": input_24,
                                "input_surface": input_surface_24,
                            },
                        )
                        input_24, input_surface_24 = output, output_surface
                    else:
                        output, output_surface = ort_session_6.run(
                            None,
                            {
                                "input": input,
                                "input_surface": input_surface,
                            },
                        )
                    input, input_surface = output, output_surface

                    # Save the results

                    if not isinstance(self.all_fields, list):
                        pl_data = output.reshape((-1, 721, 1440))
                        for data, f in zip(pl_data, fields_pl):
                            self.write(data, template=f, step=step)

                        sfc_data = output_surface.reshape((-1, 721, 1440))
                        for data, f in zip(sfc_data, fields_sfc):
                            self.write(data, template=f, step=step)
                    else:
                        surface_outputs.append(output_surface)
                        pl_outputs.append(output)
                    
                    stepper(i, step)
                
                if isinstance(self.all_fields, list):
                    surface_output = np.stack(surface_outputs).transpose((1, 0, 2, 3))
                    pl_output = np.stack(pl_outputs).transpose((1, 0, 2, 3, 4))
                    for i in range(len(self.param_sfc)):
                        data_vars[self.param_sfc[i]] = (
                            ("time", "lat", "lon"),
                            surface_output[i],
                        )
                    for i in range(len(self.param_level_pl[0])):
                        data_vars[self.param_level_pl[0][i]] = (
                            ("time", "level", "lat", "lon"),
                            pl_output[i],
                        )
                    
                    steps = np.arange(6, self.lead_time + 6, 6)    
                    times = [self.all_fields[1].time.values[0] + np.timedelta64(steps[i], 'h') for i in range(len(steps))]
                    lat, lon = self.all_fields[0].latitude.values[::-1], self.all_fields[0].longitude.values
                    saved_xarray = xr.Dataset(
                        data_vars=data_vars,
                        coords=dict(
                            lon=lon,
                            lat=lat,
                            time=times,
                            level=self.param_level_pl[1],
                        ),
                    )
                    saved_xarray = saved_xarray.reindex(level=saved_xarray.level[::-1])
                    saved_xarray = saved_xarray.rename({"level": "isobaricInhPa"})
                    start_date = self.all_fields[0].valid_time.values[0]
                    #/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/
                    """
                    name = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather/" +\
                        f"pangu_{np.datetime64(start_date, 'h')}_to_{np.datetime64(start_date + np.timedelta64(self.lead_time, 'h'), 'h')}"+\
                        f"_ldt_{self.lead_time}.nc"
                    """
                    name = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/delete_me/panguweather/" +\
                        f"pangu_{np.datetime64(start_date, 'h')}_to_{np.datetime64(start_date + np.timedelta64(self.lead_time, 'h'), 'h')}"+\
                        f"_ldt_{self.lead_time}.nc"
                        
                    LOG.info(f"Saving to {name}")
                    encoding = {}
                    encoding = {}
                    for data_var in output.data_vars:
                        encoding[data_var] = {
                        "original_shape": output[data_var].shape,
                        "_FillValue": -32767,
                        "dtype": np.float16,
                        "add_offset": output[data_var].mean().compute().values,
                        "scale_factor": output[data_var].std().compute().values / 1000, # save up to 32 std
                        # "zlib": True,
                        # "complevel": 5,
                        }
                    output.to_netcdf(name, engine="netcdf4", mode="w", encoding=encoding, compute=True)
