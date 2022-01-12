# from . import tests
from . import jmd95, natre, plot, sections  # noqa
from .eddydiff import *  # noqa

import cf_xarray as cfxr

criteria = {
    "sea_water_salinity": {
        "standard_name": "sea_water_salinity|sea_water_practical_salinity"
    }
}
cfxr.set_options(custom_criteria=criteria)
