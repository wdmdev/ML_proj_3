import numpy as np
from dataclasses import dataclass
from data_extractor import get_data

EPS = np.finfo(np.float32).eps
#Features in dataset
TARGET_FEATURE = "anmeldte tyverier/indbrud pr. 1.000 indb."
FEATURES = [
  "grundværdier pr. indb.",
  "beskatningsgrundlag pr. indb.",
  "udg. (brutto) til dagtilbud pr. indb.",
  "andel 25-64-årige uden erhvervsuddannelse",
  "andel 25-64-årige med videregående uddannelse",
  "udg. til folkeskoleområdet pr. indb.",
  "statsborgere fra ikke-vestlige lande pr. 10.000 indb.",
  "udg. til aktivering pr. 17-64/66-årig"
]

@dataclass
class Data:
  x: np.ndarray
  y: np.ndarray
  mean_x: np.ndarray = None
  mean_y: np.float = None
  std_x: np.ndarray = None
  std_y: np.float = None

def _fix_nans(data: np.ndarray):

  # Loops over municipalities
  for i in range(data.shape[0]):
    # Loops over features
    for j in range(data.shape[2]):
      dat = data[i, :, j]
      nan_idcs = np.isnan(dat)
      if nan_idcs.all():
        # If all values are nans, they are set to 0
        data[i, :, j] = 0
        continue
      elif (~nan_idcs).all():
        continue
      # Sets the first and last values to the first and last nans
      val_idcs = np.where(~np.isnan(dat))[0]
      dat[0] = dat[val_idcs[0]]
      dat[-1] = dat[val_idcs[-1]]
      nan_idcs = np.isnan(dat)
      val_idcs = np.where(~np.isnan(dat))[0]
      # Loops over years where the observation is not nan
      # The magic happens here
      for k in range(val_idcs.size-1):
        x0, x1 = val_idcs[k], val_idcs[k+1]
        y0, y1 = dat[val_idcs[k]], dat[val_idcs[k+1]]
        a = (y1 - y0) / (x1 - x0)
        b = y1 - a * x1
        x = np.arange(x0+1, x1)
        y = a * x + b
        dat[x0+1:x1] = y
      data[i, :, j] = dat
  
  return data


def create_dataset(start_year=2007, end_year=2018):

  """
  Creates a dataset with the following properties
  x is a data matrix of size nxm where m is the number of features: feature_years * n_features
  n is the number of data vectors: n_municipalities * (n_years-feature_years+1)
  y is a vector of length n with the reported number of crimes predict_years after the end of a data vector
  """

  #Getting municipality data from 2007 until 2018
  data = get_data(aarstal=[str(x) for x in range(start_year, end_year+1)], noegletal=FEATURES)
  data = _fix_nans(data)
  data_target = get_data(aarstal=[str(x) for x in range(start_year, end_year+1)], noegletal=[TARGET_FEATURE])
  data_target = _fix_nans(data_target).ravel()

  ordered = data_target.copy()
  ordered.sort()
  N = len(data_target)
  boundary = ordered[N//2]

  #Create model data
  X = []
  y = []
  for i in range(len(data)):
    for j in range(data.shape[1]-1):
      X.append(data[i, j].ravel())
  for i in range(len(X)):
    y.append(data_target[i] <= boundary)

  #Standardize
  X = np.array(X)
  y = np.array(y, dtype=np.bool)
  mean_X = X.mean(axis=0)
  std_X = X.std(axis=0) + EPS
  X = (X - mean_X) / std_X

  return X, y

if __name__ == "__main__":
  
  d, t = create_dataset()
  print(t.shape, t.dtype)

