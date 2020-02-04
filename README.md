# Project Title

Integrating and inferring parameters from the synthtetic bistable switch of Barbier, et al. 

## Description of files

* [integration_bs.py](integration_bs.py) Integration, analysis and plotting of the differential equations of the BS dynamical system.
* [fitpars.py](fitpars.py) Inference of the parameters given Flow Cytometry data. 
* [latticediff.py](latticediff.py) Definition of spatial grid to implement spatial diffusion
* [flow_load.py](flow_load.py) Import of the summarised flow data into a pandas dataframe 
* [yolcolors.py](yolcolors.py) Definition of the color palettes used

### Prerequisites

All the dependencies are listed in [requirements.txt](requirements.txt). The only nonstandard dependendcy is the PyDream MCMC

```
pip install pydream
```

## Author

* **Ruben Perez-Carrasco** - [2piruben](https://github.com/2piruben)

See also the list of contributors to the main manuscript

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

