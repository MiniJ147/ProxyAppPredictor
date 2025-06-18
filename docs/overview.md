# Overview

## Reason for Rework

Goals:  

_1: provide a cleaner working environment._  

the hold version was written in one massive single file application so naturally a lot of logic was coupled together, making it hard to iterate on. 

_2: simple to use for new users:_  

The second goal was to make the application easy for new users to come and use. This is useful as it will allow people to come and easily re-create results from papers or run there own experiments.

_3: easy to scale_  

The third and last goal was to make the application easy to scale, meaning stream lining the process of create new applications or predictors in order to run new experiments quicker.  

## Design Overview

### Purpose
The purpose of this section is to give the necessary knowledge to users who are interested in tinkering with the codebase for either tweaking experiments or adding new applications/predictors.  

## Adding Apps

### Params
For adding custom parameters look at params.json  
_This code is only useful if you are automatically generating input files..._
```json
//params.json
{
    "app_name":{
        "default":{
            "param_name": "default_val"   
        },
        "range":{
            // (optional)
            "param_name": ["range","of","values"] // param_name must match to default
        }
    }
}
```

default_param will be the value picked if you do not decide to randomize it  
range_param (if present) will then randomly pick one of the values in the list  

### New App Class 

For adding a new application just inherit from the base class named App

Then Feel free to override any functions provided  

```python
class App:
    #code...

class NewApp(App): # inherit
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("new app name", pred_col, test_file_path)
    
    # override any function that needs its behavior changed...
```

### App Functions

```python
# (base class)
# should always be called first with super().parse() as it handles the necessary parsing of the csv / converting values for the simulator to understand  
# from there you will get returned an x, y which you can modify if need in your override for your application 
def parse(self) -> X, y:


```