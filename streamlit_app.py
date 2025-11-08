import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings
from sklearn.metrics import mean_absolute_percentage_error
as calculate_mape
 
# Suppress warnings from statsmodels, which are common in
# Streamlit environments
warnings.filterwarnings("ignore")
 
# --- 1. Data Loading and Initial Preprocessing ---
# Data snippet provided by the user from the uploaded file
CSV_CONTENT = """Barangay,Year,Quarter,Period,Copra_Production
(MT),Farmgate Price (PHP/kg),Millgate Price (PHP/kg),Area (hectares)
Poblacion,2015,Q1,2015-01-01,32.17,24.50,28.50,39
Poblacion,2015,Q2,2015-04-01,32.15,23.00,27.00,39
Poblacion,2015,Q3,2015-07-01,32.18,21.00,25.00,39
Poblacion,2015,Q4,2015-10-01,33.13,16.50,20.50,39
Poblacion,2016,Q1,2016-01-01,33.00,18.00,22.00,40
Poblacion,2016,Q2,2016-04-01,33.00,20.50,24.50,40
Poblacion,2016,Q3,2016-07-01,32.00,23.00,27.00,40
Poblacion,2016,Q4,2016-10-01,33.18,26.50,30.50,40
Poblacion,2017,Q1,2017-01-01,32.00,42.00,46.00,40
Poblacion,2017,Q2,2017-04-01,32.00,45.00,49.00,40
Poblacion,2017,Q3,2017-07-01,31.00,46.50,50.50,40
Poblacion,2017,Q4,2017-10-01,33.00,46.00,50.00,40
Poblacion,2018,Q1,2018-01-01,32.00,38.00,42.00,40
Poblacion,2018,Q2,2018-04-01,32.00,28.00,32.00,40
Poblacion,2018,Q3,2018-07-01,33.00,22.00,26.00,40
Poblacion,2018,Q4,2018-10-01,31.00,18.00,22.00,40
Poblacion,2019,Q1,2019-01-01,7.17,14.00,18.00,41
Poblacion,2019,Q2,2019-04-01,7.15,14.50,18.50,41
Poblacion,2019,Q3,2019-07-01,7.18,15.00,19.00,41
Poblacion,2019,Q4,2019-10-01,7.12,16.50,20.50,41
Poblacion,2020,Q1,2020-01-01,7.09,20.00,24.00,41
Poblacion,2020,Q2,2020-04-01,7.10,24.00,28.00,41
Poblacion,2020,Q3,2020-07-01,7.02,25.50,29.50,41
Poblacion,2020,Q4,2020-10-01,7.30,28.50,32.50,41
Poblacion,2021,Q1,2021-01-01,7.40,31.00,35.00,41
Poblacion,2021,Q2,2021-04-01,7.10,34.00,38.00,41
Poblacion,2021,Q3,2021-07-01,7.30,35.50,39.50,41
Poblacion,2021,Q4,2021-10-01,7.40,36.50,40.50,41
Poblacion,2022,Q1,2022-01-01,14.58,40.00,44.00,42
Poblacion,2022,Q2,2022-04-01,14.60,48.00,57.10,42
Poblacion,2022,Q3,2022-07-01,14.49,36.00,40.00,42
Poblacion,2022,Q4,2022-10-01,14.57,28.00,32.00,42
Poblacion,2023,Q1,2023-01-01,35.61,23.85,34.00,43
Poblacion,2023,Q2,2023-04-01,35.60,25.00,35.00,43
Poblacion,2023,Q3,2023-07-01,35.70,27.00,37.00,43
Poblacion,2023,Q4,2023-10-01,35.65,29.00,39.00,43
Poblacion,2024,Q1,2024-01-01,38.40,32.00,42.00,45
Poblacion,2024,Q2,2024-04-01,38.10,34.00,44.00,45
Poblacion,2024,Q3,2024-07-01,38.90,34.07,44.18,45
Poblacion,2024,Q4,2024-10-01,39.20,40.00,50.00,45
Poblacion,2025,Q1,2025-01-01,40.42,48.64,52.12,46
Poblacion,2025,Q2,2025-04-01,40.40,64.55,81.00,46
Poblacion,2025,Q3,2025-07-01,40.70,56.79,72.70,46
Bunawan Brook,2015,Q1,2015-01-01,417.45,24.50,28.50,425
Bunawan Brook,2015,Q2,2015-04-01,415.12,23.00,27.00,425
Bunawan Brook,2015,Q3,2015-07-01,412.80,21.00,25.00,425
Bunawan Brook,2015,Q4,2015-10-01,416.21,16.50,20.50,425
Bunawan Brook,2016,Q1,2016-01-01,419.92,18.00,22.00,420
Bunawan Brook,2016,Q2,2016-04-01,419.18,20.50,24.50,420
Bunawan Brook,2016,Q3,2016-07-01,419.71,23.00,27.00,420
Bunawan Brook,2016,Q4,2016-10-01,420.28,26.50,30.50,420
Bunawan Brook,2017,Q1,2017-01-01,411.20,42.00,46.00,420
Bunawan Brook,2017,Q2,2017-04-01,411.30,45.00,49.00,420
Bunawan Brook,2017,Q3,2017-07-01,411.50,46.50,50.50,420
Bunawan Brook,2017,Q4,2017-10-01,411.20,46.00,50.00,420
Bunawan Brook,2018,Q1,2018-01-01,416.00,38.00,42.00,420
Bunawan Brook,2018,Q2,2018-04-01,415.00,28.00,32.00,420
Bunawan Brook,2018,Q3,2018-07-01,416.00,22.00,26.00,420
Bunawan Brook,2018,Q4,2018-10-01,417.00,18.00,22.00,420
Bunawan Brook,2019,Q1,2019-01-01,89.14,14.00,18.00,430
Bunawan Brook,2019,Q2,2019-04-01,89.50,14.50,18.50,430
Bunawan Brook,2019,Q3,2019-07-01,89.90,15.00,19.00,430
Bunawan Brook,2019,Q4,2019-10-01,89.15,16.50,20.50,430
Bunawan Brook,2020,Q1,2020-01-01,93.83,20.00,24.00,430
Bunawan Brook,2020,Q2,2020-04-01,93.80,24.00,28.00,430
Bunawan Brook,2020,Q3,2020-07-01,93.90,25.50,29.50,430
Bunawan Brook,2020,Q4,2020-10-01,93.01,28.50,32.50,430
Bunawan Brook,2021,Q1,2021-01-01,95.80,31.00,35.00,430
Bunawan Brook,2021,Q2,2021-04-01,95.70,34.00,38.00,430
Bunawan Brook,2021,Q3,2021-07-01,95.80,35.50,39.50,430
Bunawan Brook,2021,Q4,2021-10-01,95.60,36.50,40.50,430
Bunawan Brook,2022,Q1,2022-01-01,178.40,40.00,44.00,440
Bunawan Brook,2022,Q2,2022-04-01,178.50,48.00,57.10,440
Bunawan Brook,2022,Q3,2022-07-01,178.60,36.00,40.00,440
Bunawan Brook,2022,Q4,2022-10-01,178.50,28.00,32.00,440
Bunawan Brook,2023,Q1,2023-01-01,1147.03,23.85,34.00,450
Bunawan Brook,2023,Q2,2023-04-01,1147.00,25.00,35.00,450
Bunawan Brook,2023,Q3,2023-07-01,1148.00,27.00,37.00,450
Bunawan Brook,2023,Q4,2023-10-01,1147.00,29.00,39.00,450
Bunawan Brook,2024,Q1,2024-01-01,560.00,32.00,42.00,460
Bunawan Brook,2024,Q2,2024-04-01,561.00,34.00,44.00,460
Bunawan Brook,2024,Q3,2024-07-01,560.00,34.07,44.18,460
Bunawan Brook,2024,Q4,2024-10-01,562.00,40.00,50.00,460
Bunawan Brook,2025,Q1,2025-01-01,584.92,48.64,52.12,470
Bunawan Brook,2025,Q2,2025-04-01,585.00,64.55,81.00,470
Bunawan Brook,2025,Q3,2025-07-01,586.00,56.79,72.70,470
Consuelo,2015,Q1,2015-01-01,115.50,24.50,28.50,119
Consuelo,2015,Q2,2015-04-01,115.51,23.00,27.00,119
Consuelo,2015,Q3,2015-07-01,114.51,21.00,25.00,119
Consuelo,2015,Q4,2015-10-01,115.98,16.50,20.50,119
Consuelo,2016,Q1,2016-01-01,118.80,18.00,22.00,120
Consuelo,2016,Q2,2016-04-01,118.71,20.50,24.50,120
Consuelo,2016,Q3,2016-07-01,118.25,23.00,27.00,120
Consuelo,2016,Q4,2016-10-01,118.34,26.50,30.50,120
Consuelo,2017,Q1,2017-01-01,118.40,42.00,46.00,120
Consuelo,2017,Q2,2017-04-01,118.30,45.00,49.00,120
Consuelo,2017,Q3,2017-07-01,118.70,46.50,50.50,120
Consuelo,2017,Q4,2017-10-01,118.20,46.00,50.00,120
Consuelo,2018,Q1,2018-01-01,120.80,38.00,42.00,120
Consuelo,2018,Q2,2018-04-01,120.00,28.00,32.00,120
Consuelo,2018,Q3,2018-07-01,119.50,22.00,26.00,120
Consuelo,2018,Q4,2018-10-01,118.00,18.00,22.00,120
Consuelo,2019,Q1,2019-01-01,26.72,14.00,18.00,122
Consuelo,2019,Q2,2019-04-01,26.71,14.50,18.50,122
Consuelo,2019,Q3,2019-07-01,26.80,15.00,19.00,122
Consuelo,2019,Q4,2019-10-01,26.79,16.50,20.50,122
Consuelo,2020,Q1,2020-01-01,25.21,20.00,24.00,122
Consuelo,2020,Q2,2020-04-01,25.15,24.00,28.00,122
Consuelo,2020,Q3,2020-07-01,25.20,25.50,29.50,122
Consuelo,2020,Q4,2020-10-01,25.21,28.50,32.50,122
Consuelo,2021,Q1,2021-01-01,25.84,31.00,35.00,122
Consuelo,2021,Q2,2021-04-01,25.80,34.00,38.00,122
Consuelo,2021,Q3,2021-07-01,25.00,35.50,39.50,122
Consuelo,2021,Q4,2021-10-01,25.90,36.50,40.50,122
Consuelo,2022,Q1,2022-01-01,48.70,40.00,44.00,123
Consuelo,2022,Q2,2022-04-01,48.70,48.00,57.10,123
Consuelo,2022,Q3,2022-07-01,48.90,36.00,40.00,123
Consuelo,2022,Q4,2022-10-01,48.30,28.00,32.00,123
Consuelo,2023,Q1,2023-01-01,392.53,23.85,34.00,124
Consuelo,2023,Q2,2023-04-01,392.00,25.00,35.00,124
Consuelo,2023,Q3,2023-07-01,390.00,27.00,37.00,124
Consuelo,2023,Q4,2023-10-01,391.00,29.00,39.00,124
Consuelo,2024,Q1,2024-01-01,202.40,32.00,42.00,125
Consuelo,2024,Q2,2024-04-01,202.40,34.00,44.00,125
Consuelo,2024,Q3,2024-07-01,203.00,34.07,44.18,125
Consuelo,2024,Q4,2024-10-01,203.00,40.00,50.00,125
Consuelo,2025,Q1,2025-01-01,228.94,48.64,52.12,126
Consuelo,2025,Q2,2025-04-01,228.90,64.55,81.00,126
Consuelo,2025,Q3,2025-07-01,229.00,56.79,72.70,126
Libertad,2015,Q1,2015-01-01,16.50,24.50,28.50,17
Libertad,2015,Q2,2015-04-01,16.30,23.00,27.00,17
Libertad,2015,Q3,2015-07-01,16.10,21.00,25.00,17
Libertad,2015,Q4,2015-10-01,16.12,16.50,20.50,17
Libertad,2016,Q1,2016-01-01,16.50,18.00,22.00,16
Libertad,2016,Q2,2016-04-01,16.30,20.50,24.50,16
Libertad,2016,Q3,2016-07-01,16.80,23.00,27.00,16
Libertad,2016,Q4,2016-10-01,16.29,26.50,30.50,16
Libertad,2017,Q1,2017-01-01,16.00,42.00,46.00,16
Libertad,2017,Q2,2017-04-01,16.00,45.00,49.00,16
Libertad,2017,Q3,2017-07-01,15.80,46.50,50.50,16
Libertad,2017,Q4,2017-10-01,16.30,46.00,50.00,16
Libertad,2018,Q1,2018-01-01,16.00,38.00,42.00,16
Libertad,2018,Q2,2018-04-01,16.00,28.00,32.00,16
Libertad,2018,Q3,2018-07-01,15.00,22.00,26.00,16
Libertad,2018,Q4,2018-10-01,15.30,18.00,22.00,16
Libertad,2019,Q1,2019-01-01,3.50,14.00,18.00,17
Libertad,2019,Q2,2019-04-01,3.30,14.50,18.50,17
Libertad,2019,Q3,2019-07-01,3.50,15.00,19.00,17
Libertad,2019,Q4,2019-10-01,3.50,16.50,20.50,17
Libertad,2020,Q1,2020-01-01,3.23,20.00,24.00,17
Libertad,2020,Q2,2020-04-01,3.21,24.00,28.00,17
Libertad,2020,Q3,2020-07-01,3.25,25.50,29.50,17
Libertad,2020,Q4,2020-10-01,3.25,28.50,32.50,17
Libertad,2021,Q1,2021-01-01,3.15,31.00,35.00,17
Libertad,2021,Q2,2021-04-01,3.15,34.00,38.00,17
Libertad,2021,Q3,2021-07-01,3.15,35.50,39.50,17
Libertad,2021,Q4,2021-10-01,3.15,36.50,40.50,17
Libertad,2022,Q1,2022-01-01,5.83,40.00,44.00,18
Libertad,2022,Q2,2022-04-01,5.81,48.00,57.10,18
Libertad,2022,Q3,2022-07-01,5.79,36.00,40.00,18
Libertad,2022,Q4,2022-10-01,5.85,28.00,32.00,18
Libertad,2023,Q1,2023-01-01,34.16,23.85,34.00,19
Libertad,2023,Q2,2023-04-01,34.17,25.00,35.00,19
Libertad,2023,Q3,2023-07-01,34.18,27.00,37.00,19
Libertad,2023,Q4,2023-10-01,35.10,29.00,39.00,19
Libertad,2024,Q1,2024-01-01,16.80,32.00,42.00,20
Libertad,2024,Q2,2024-04-01,16.90,34.00,44.00,20
Libertad,2024,Q3,2024-07-01,16.70,34.07,44.18,20
Libertad,2024,Q4,2024-10-01,17.10,40.00,50.00,20
Libertad,2025,Q1,2025-01-01,17.66,48.64,52.12,21
Libertad,2025,Q2,2025-04-01,17.60,64.55,81.00,21
Libertad,2025,Q3,2025-07-01,17.80,56.79,72.70,21
Imelda,2015,Q1,2015-01-01,74.25,24.50,28.50,75
Imelda,2015,Q2,2015-04-01,74.29,23.00,27.00,75
Imelda,2015,Q3,2015-07-01,73.68,21.00,25.00,75
Imelda,2015,Q4,2015-10-01,74.57,16.50,20.50,75
Imelda,2016,Q1,2016-01-01,78.37,18.00,22.00,79
Imelda,2016,Q2,2016-04-01,78.32,20.50,24.50,79
Imelda,2016,Q3,2016-07-01,79.15,23.00,27.00,79
Imelda,2016,Q4,2016-10-01,79.36,26.50,30.50,79
Imelda,2017,Q1,2017-01-01,78.40,42.00,46.00,79
Imelda,2017,Q2,2017-04-01,78.50,45.00,49.00,79
Imelda,2017,Q3,2017-07-01,77.90,46.50,50.50,79
Imelda,2017,Q4,2017-10-01,78.30,46.00,50.00,79
Imelda,2018,Q1,2018-01-01,80.80,38.00,42.00,79
Imelda,2018,Q2,2018-04-01,80.30,28.00,32.00,79
Imelda,2018,Q3,2018-07-01,79.50,22.00,26.00,79
Imelda,2018,Q4,2018-10-01,79.90,18.00,22.00,79
Imelda,2019,Q1,2019-01-01,18.10,14.00,18.00,81
Imelda,2019,Q2,2019-04-01,18.20,14.50,18.50,81
Imelda,2019,Q3,2019-07-01,18.10,15.00,19.00,81
Imelda,2019,Q4,2019-10-01,18.50,16.50,20.50,81
Imelda,2020,Q1,2020-01-01,17.17,20.00,24.00,81
Imelda,2020,Q2,2020-04-01,17.15,24.00,28.00,81
Imelda,2020,Q3,2020-07-01,17.21,25.50,29.50,81
Imelda,2020,Q4,2020-10-01,17.30,28.50,32.50,81
Imelda,2021,Q1,2021-01-01,17.64,31.00,35.00,81
Imelda,2021,Q2,2021-04-01,17.60,34.00,38.00,81
Imelda,2021,Q3,2021-07-01,17.50,35.50,39.50,81
Imelda,2021,Q4,2021-10-01,17.70,36.50,40.50,81
Imelda,2022,Q1,2022-01-01,32.95,40.00,44.00,82
Imelda,2022,Q2,2022-04-01,32.90,48.00,57.10,82
Imelda,2022,Q3,2022-07-01,32.96,36.00,40.00,82
Imelda,2022,Q4,2022-10-01,32.89,28.00,32.00,82
Imelda,2023,Q1,2023-01-01,236.50,23.85,34.00,83
Imelda,2023,Q2,2023-04-01,236.50,25.00,35.00,83
Imelda,2023,Q3,2023-07-01,236.70,27.00,37.00,83
Imelda,2023,Q4,2023-10-01,236.90,29.00,39.00,83
Imelda,2024,Q1,2024-01-01,115.36,32.00,42.00,85
Imelda,2024,Q2,2024-04-01,115.50,34.00,44.00,85
Imelda,2024,Q3,2024-07-01,115.60,34.07,44.18,85
Imelda,2024,Q4,2024-10-01,116.30,40.00,50.00,85
Imelda,2025,Q1,2025-01-01,120.65,48.64,52.12,86
Imelda,2025,Q2,2025-04-01,120.00,64.55,81.00,86
Imelda,2025,Q3,2025-07-01,121.50,56.79,72.70,86
Mambalili,2015,Q1,2015-01-01,12.37,24.50,28.50,13
Mambalili,2015,Q2,2015-04-01,12.31,23.00,27.00,13
Mambalili,2015,Q3,2015-07-01,12.52,21.00,25.00,13
Mambalili,2015,Q4,2015-10-01,12.38,16.50,20.50,13
Mambalili,2016,Q1,2016-01-01,12.37,18.00,22.00,12
Mambalili,2016,Q2,2016-04-01,11.98,20.50,24.50,12
Mambalili,2016,Q3,2016-07-01,11.81,23.00,27.00,12
Mambalili,2016,Q4,2016-10-01,12.68,26.50,30.50,12
Mambalili,2017,Q1,2017-01-01,12.00,42.00,46.00,12
Mambalili,2017,Q2,2017-04-01,12.00,45.00,49.00,12
Mambalili,2017,Q3,2017-07-01,13.00,46.50,50.50,12
Mambalili,2017,Q4,2017-10-01,12.00,46.00,50.00,12
Mambalili,2018,Q1,2018-01-01,12.00,38.00,42.00,12
Mambalili,2018,Q2,2018-04-01,12.10,28.00,32.00,12
Mambalili,2018,Q3,2018-07-01,12.50,22.00,26.00,12
Mambalili,2018,Q4,2018-10-01,12.90,18.00,22.00,12
Mambalili,2019,Q1,2019-01-01,2.86,14.00,18.00,13
Mambalili,2019,Q2,2019-04-01,2.83,14.50,18.50,13
Mambalili,2019,Q3,2019-07-01,2.86,15.00,19.00,13
Mambalili,2019,Q4,2019-10-01,2.70,16.50,20.50,13
Mambalili,2020,Q1,2020-01-01,2.60,20.00,24.00,13
Mambalili,2020,Q2,2020-04-01,2.10,24.00,28.00,13
Mambalili,2020,Q3,2020-07-01,2.30,25.50,29.50,13
Mambalili,2020,Q4,2020-10-01,2.50,28.50,32.50,13
Mambalili,2021,Q1,2021-01-01,2.68,31.00,35.00,13
Mambalili,2021,Q2,2021-04-01,2.70,34.00,38.00,13
Mambalili,2021,Q3,2021-07-01,2.60,35.50,39.50,13
Mambalili,2021,Q4,2021-10-01,2.64,36.50,40.50,13
Mambalili,2022,Q1,2022-01-01,4.96,40.00,44.00,14
Mambalili,2022,Q2,2022-04-01,4.95,48.00,57.10,14
Mambalili,2022,Q3,2022-07-01,4.90,36.00,40.00,14
Mambalili,2022,Q4,2022-10-01,4.70,28.00,32.00,14
Mambalili,2023,Q1,2023-01-01,27.92,23.85,34.00,15
Mambalili,2023,Q2,2023-04-01,27.90,25.00,35.00,15
Mambalili,2023,Q3,2023-07-01,28.20,27.00,37.00,15
Mambalili,2023,Q4,2023-10-01,27.60,29.00,39.00,15
Mambalili,2024,Q1,2024-01-01,13.60,32.00,42.00,16
Mambalili,2024,Q2,2024-04-01,13.50,34.00,44.00,16
Mambalili,2024,Q3,2024-07-01,13.80,34.07,44.18,16
Mambalili,2024,Q4,2024-10-01,14.30,40.00,50.00,16
Mambalili,2025,Q1,2025-01-01,14.29,48.64,52.12,17
Mambalili,2025,Q2,2025-04-01,14.30,64.55,81.00,17
Mambalili,2025,Q3,2025-07-01,14.40,56.79,72.70,17
San Andres,2015,Q1,2015-01-01,74.25,24.50,28.50,75
San Andres,2015,Q2,2015-04-01,74.21,23.00,27.00,75
San Andres,2015,Q3,2015-07-01,74.30,21.00,25.00,75
San Andres,2015,Q4,2015-10-01,75.22,16.50,20.50,75
San Andres,2016,Q1,2016-01-01,76.72,18.00,22.00,77
San Andres,2016,Q2,2016-04-01,76.82,20.50,24.50,77
San Andres,2016,Q3,2016-07-01,77.31,23.00,27.00,77
San Andres,2016,Q4,2016-10-01,78.10,26.50,30.50,77
San Andres,2017,Q1,2017-01-01,75.20,42.00,46.00,77
San Andres,2017,Q2,2017-04-01,75.00,45.00,49.00,77
San Andres,2017,Q3,2017-07-01,74.00,46.50,50.50,77
San Andres,2017,Q4,2017-10-01,73.00,46.00,50.00,77
San Andres,2018,Q1,2018-01-01,75.20,38.00,42.00,77
San Andres,2018,Q2,2018-04-01,75.10,28.00,32.00,77
San Andres,2018,Q3,2018-07-01,74.80,22.00,26.00,77
San Andres,2018,Q4,2018-10-01,74.40,18.00,22.00,77
San Andres,2019,Q1,2019-01-01,16.43,14.00,18.00,78
San Andres,2019,Q2,2019-04-01,16.40,14.50,18.50,78
San Andres,2019,Q3,2019-07-01,16.42,15.00,19.00,78
San Andres,2019,Q4,2019-10-01,16.47,16.50,20.50,78
San Andres,2020,Q1,2020-01-01,16.07,20.00,24.00,78
San Andres,2020,Q2,2020-04-01,16.10,24.00,28.00,78
San Andres,2020,Q3,2020-07-01,16.05,25.50,29.50,78
San Andres,2020,Q4,2020-10-01,16.20,28.00,32.50,78
San Andres,2021,Q1,2021-01-01,17.02,31.00,35.00,78
San Andres,2021,Q2,2021-04-01,17.01,34.00,38.00,78
San Andres,2021,Q3,2021-07-01,17.11,35.50,39.50,78
San Andres,2021,Q4,2021-10-01,17.05,36.50,40.50,78
San Andres,2022,Q1,2022-01-01,31.78,40.00,44.00,79
San Andres,2022,Q2,2022-04-01,31.77,48.00,57.10,79
San Andres,2022,Q3,2022-07-01,31.75,36.00,40.00,79
San Andres,2022,Q4,2022-10-01,31.80,28.00,32.00,79
San Andres,2023,Q1,2023-01-01,117.37,23.85,34.00,80
San Andres,2023,Q2,2023-04-01,117.30,25.00,35.00,80
San Andres,2023,Q3,2023-07-01,117.50,27.00,37.00,80
San Andres,2023,Q4,2023-10-01,117.80,29.00,39.00,80
San Andres,2024,Q1,2024-01-01,86.64,32.00,42.00,82
San Andres,2024,Q2,2024-04-01,86.60,34.00,44.00,82
San Andres,2024,Q3,2024-07-01,86.80,34.07,44.18,82
San Andres,2024,Q4,2024-10-01,86.70,40.00,50.00,82
San Andres,2025,Q1,2025-01-01,89.80,48.64,52.12,83
San Andres,2025,Q2,2025-04-01,89.70,64.55,81.00,83
San Andres,2025,Q3,2025-07-01,89.90,56.79,72.70,83
San Teodoro,2015,Q1,2015-01-01,33.00,24.50,28.50,34
San Teodoro,2015,Q2,2015-04-01,34.12,23.00,27.00,34
San Teodoro,2015,Q3,2015-07-01,33.95,21.00,25.00,34
San Teodoro,2015,Q4,2015-10-01,34.67,16.50,20.50,34
San Teodoro,2016,Q1,2016-01-01,34.65,18.00,22.00,33
San Teodoro,2016,Q2,2016-04-01,33.12,20.50,24.50,33
San Teodoro,2016,Q3,2016-07-01,32.80,23.00,27.00,33
San Teodoro,2016,Q4,2016-10-01,33.17,26.50,30.50,33
San Teodoro,2017,Q1,2017-01-01,33.60,42.00,46.00,33
San Teodoro,2017,Q2,2017-04-01,33.00,45.00,49.00,33
San Teodoro,2017,Q3,2017-07-01,34.00,46.50,50.50,33
San Teodoro,2017,Q4,2017-10-01,32.00,46.00,50.00,33
San Teodoro,2018,Q1,2018-01-01,33.60,38.00,42.00,33
San Teodoro,2018,Q2,2018-04-01,33.90,28.00,32.00,33
San Teodoro,2018,Q3,2018-07-01,33.51,22.00,26.00,33
San Teodoro,2018,Q4,2018-10-01,32.68,18.00,22.00,33
San Teodoro,2019,Q1,2019-01-01,7.48,14.00,18.00,35
San Teodoro,2019,Q2,2019-04-01,7.48,14.50,18.50,35
San Teodoro,2019,Q3,2019-07-01,7.60,15.00,19.00,35
San Teodoro,2019,Q4,2019-10-01,7.59,16.50,20.50,35
San Teodoro,2020,Q1,2020-01-01,6.93,20.00,24.00,35
San Teodoro,2020,Q2,2020-04-01,6.91,24.00,28.00,35
San Teodoro,2020,Q3,2020-07-01,6.90,25.50,29.50,35
San Teodoro,2020,Q4,2020-10-01,6.70,28.50,32.50,35
San Teodoro,2021,Q1,2021-01-01,7.41,31.00,35.00,35
San Teodoro,2021,Q2,2021-04-01,7.40,34.00,38.00,35
San Teodoro,2021,Q3,2021-07-01,7.44,35.50,39.50,35
San Teodoro,2021,Q4,2021-10-01,7.50,36.50,40.50,35
San Teodoro,2022,Q1,2022-01-01,13.70,40.00,44.00,36
San Teodoro,2022,Q2,2022-04-01,13.70,48.00,57.10,36
San Teodoro,2022,Q3,2022-07-01,13.69,36.00,40.00,36
San Teodoro,2022,Q4,2022-10-01,13.80,28.00,32.00,36
San Teodoro,2023,Q1,2023-01-01,80.18,23.85,34.00,37
San Teodoro,2023,Q2,2023-04-01,80.20,25.00,35.00,37
San Teodoro,2023,Q3,2023-07-01,80.60,27.00,37.00,37
San Teodoro,2023,Q4,2023-10-01,80.20,29.00,39.00,37
San Teodoro,2024,Q1,2024-01-01,39.60,32.00,42.00,39
San Teodoro,2024,Q2,2024-04-01,39.50,34.00,44.00,39
San Teodoro,2024,Q3,2024-07-01,39.80,34.07,44.18,39
San Teodoro,2024,Q4,2024-10-01,40.00,40.00,50.00,39
San Teodoro,2025,Q1,2025-01-01,41.51,48.64,52.12,40
San Teodoro,2025,Q2,2025-04-01,41.55,64.55,81.00,40
San Teodoro,2025,Q3,2025-07-01,41.80,56.79,72.70,40
Nueva Era,2015,Q1,2015-01-01,11.55,24.50,28.50,12
Nueva Era,2015,Q2,2015-04-01,11.51,23.00,27.00,12
Nueva Era,2015,Q3,2015-07-01,11.59,21.00,25.00,12
Nueva Era,2015,Q4,2015-10-01,11.78,16.50,20.50,12
Nueva Era,2016,Q1,2016-01-01,11.55,18.00,22.00,13
Nueva Era,2016,Q2,2016-04-01,11.57,20.50,24.50,13
Nueva Era,2016,Q3,2016-07-01,11.71,23.00,27.00,13
Nueva Era,2016,Q4,2016-10-01,12.13,26.50,30.50,13
Nueva Era,2017,Q1,2017-01-01,11.20,42.00,46.00,14
Nueva Era,2017,Q2,2017-04-01,11.30,45.00,49.00,14
Nueva Era,2017,Q3,2017-07-01,11.00,46.50,50.50,14
Nueva Era,2017,Q4,2017-10-01,11.10,46.00,50.00,14
Nueva Era,2018,Q1,2018-01-01,11.20,38.00,42.00,14
Nueva Era,2018,Q2,2018-04-01,11.00,28.00,32.00,14
Nueva Era,2018,Q3,2018-07-01,11.30,22.00,26.00,14
Nueva Era,2018,Q4,2018-10-01,12.10,18.00,22.00,14
Nueva Era,2019,Q1,2019-01-01,2.56,14.00,18.00,15
Nueva Era,2019,Q2,2019-04-01,2.55,14.50,18.50,15
Nueva Era,2019,Q3,2019-07-01,2.58,15.00,19.00,15
Nueva Era,2019,Q4,2019-10-01,2.49,16.50,20.50,15
Nueva Era,2020,Q1,2020-01-01,2.36,20.00,24.00,15
Nueva Era,2020,Q2,2020-04-01,2.60,24.00,28.00,15
Nueva Era,2020,Q3,2020-07-01,2.35,25.50,29.50,15
Nueva Era,2020,Q4,2020-10-01,2.40,28.50,32.50,15
Nueva Era,2021,Q1,2021-01-01,2.36,31.00,35.00,15
Nueva Era,2021,Q2,2021-04-01,2.31,34.00,38.00,15
Nueva Era,2021,Q3,2021-07-01,2.37,35.50,39.50,15
Nueva Era,2021,Q4,2021-10-01,2.39,36.50,40.50,15
Nueva Era,2022,Q1,2022-01-01,4.25,40.00,44.00,16
Nueva Era,2022,Q2,2022-04-01,4.21,48.00,57.10,16
Nueva Era,2022,Q3,2022-07-01,4.26,36.00,40.00,16
Nueva Era,2022,Q4,2022-10-01,4.20,28.00,32.00,16
Nueva Era,2023,Q1,2023-01-01,24.63,23.85,34.00,17
Nueva Era,2023,Q2,2023-04-01,24.64,25.00,35.00,17
Nueva Era,2023,Q3,2023-07-01,24.70,27.00,37.00,17
Nueva Era,2023,Q4,2023-10-01,24.80,29.00,39.00,17
Nueva Era,2024,Q1,2024-01-01,12.00,32.00,42.00,19
Nueva Era,2024,Q2,2024-04-01,12.00,34.00,44.00,19
Nueva Era,2024,Q3,2024-07-01,13.00,34.07,44.18,19
Nueva Era,2024,Q4,2024-10-01,12.00,40.00,50.00,19
Nueva Era,2025,Q1,2025-01-01,12.37,48.64,52.12,20
Nueva Era,2025,Q2,2025-04-01,12.40,64.55,81.00,20
Nueva Era,2025,Q3,2025-07-01,12.50,56.79,72.70,20"""
 
@st.cache_data
def load_data():
   """Loads and preprocesses the Copra Production
data."""
   df = pd.read_csv(io.StringIO(CSV_CONTENT))
 
   # Convert 'Period'
# to datetime objects and set as index
   df['Period'] = pd.to_datetime(df['Period'])
   df['Year'] = df['Period'].dt.year
   df['Quarter'] = 'Q' + df['Period'].dt.quarter.astype(str)
   
   # Handle any
# potential missing values by filling with the mean of the column
   # We do this
# before using the data in case of dynamic row additions.
   df.fillna(df.mean(numeric_only=True), inplace=True) 
   
   return df
 
def initialize_session_data():
   """Initializes the data into Streamlit session state if
not already present."""
   if 'df_data' not in st.session_state:
       st.session_state['df_data'] = load_data()
 
 
# --- 2. ARIMA Forecasting Helper Function ---
 
def _fit_and_forecast_single_series(data_series,
forecast_end_year, series_name):
   """
   Fits an ARIMA
# model for a single time series, calculates MAPE, and forecasts.
    
   Args:
       data_series
# (pd.Series): The time series data.
       forecast_end_year (int): The last year to forecast to (e.g., 2035).
       series_name
# (str): The name of the series for context.
        
   Returns:
       tuple:
# (pd.Series, str, str) -> (Forecast Values Series, Model Summary Text, MAPE
# String)
   """
   if data_series.empty or len(data_series) < 5:
       # FIX: Combined the f-string onto a single line to resolve the SyntaxError
       return None, f"Error: Insufficient data for {series_name} (need at least 5 quarters).", "N/A"
        
   n_test = 4
   mape_str = "N/A (Not enough data points for validation)"
    
   try:
       # 1. Backtest
# Split for MAPE Calculation (using last 4 quarters for testing)
       if len(data_series) > n_test:
           train_data = data_series[:-n_test]
           test_data = data_series[-n_test:]
 
           #
# Temporarily fit model on training data for evaluation
           # Using
# freq='QS-JAN' assumes quarterly data starting in Jan (Q1, Q2, Q3, Q4)
           # ARIMA
# order (1, 1, 0) is a simple model for demonstration
           model_train = ARIMA(train_data, order=(1, 1, 0), freq='QS-JAN')
           model_fit_train = model_train.fit()
            
           # Predict
# the test period
           test_forecast = model_fit_train.get_forecast(steps=n_test)
           test_pred = test_forecast.predicted_mean
            
           #
# Calculate MAPE
           mape_value = calculate_mape(test_data.values, test_pred.values) * 100
           mape_str = f"{mape_value:.2f}% "
            
       # 2. Main
# Forecast: Fit model on ALL available historical data
       model_full = ARIMA(data_series, order=(1, 1, 0), freq='QS-JAN')
       model_fit_full = model_full.fit()
        
       # Determine
# the start date for forecasting
       start_date = data_series.index[-1] + DateOffset(months=3)
        
       # Create the
# future date range (Quarterly Start frequency)
       future_dates = pd.date_range(start=start_date, end=f'{forecast_end_year}-10-01', freq='QS')
        
        # Generate the forecast
       forecast = model_fit_full.get_forecast(steps=len(future_dates))
       forecast_values = forecast.predicted_mean
       forecast_values.index = future_dates
        
       return forecast_values, model_fit_full.summary().as_text(), mape_str
        
   except Exception as e:
       # Print the
# error to the console for debugging but return a user-friendly message
       print(f"ARIMA Model Error for {series_name}: {e}")
       return None, f"ARIMA Model Error for {series_name}: {e}", "N/A"
 
# --- 3. ARIMA Forecasting Pipeline (Cached) ---
 
@st.cache_data
def arima_forecast(ts_production, ts_farmgate, ts_millgate,
forecast_end_year, last_historical_date):
   """
   Runs ARIMA
# forecasting on Copra Production, Farmgate Price, and Millgate Price.
    
   Returns:
       tuple:
# (df_combined_plot, df_combined_forecast, mape_metrics, model_summaries)
       df_combined_plot: DataFrame containing both historical and forecast data
# for plotting.
       df_combined_forecast: DataFrame containing only the forecast data.
       mape_metrics:
# Dictionary of MAPE strings for each metric.
       model_summaries: Dictionary of model summary texts for each metric.
   """
    
   # Define series to
# process
   series_map = {
       'Copra_Production (MT)': ts_production,
       'Farmgate Price (PHP/kg)': ts_farmgate,
       'Millgate Price (PHP/kg)': ts_millgate
   }
    
   # Prepare
# containers for results
   forecast_results = {}
   mape_metrics = {}
   model_summaries = {}
    
   # 1. Run forecast
# for each series
   for name, series in series_map.items():
       forecast_series, summary, mape = _fit_and_forecast_single_series(
           series, 
           forecast_end_year,
            name
        )
        
       if forecast_series is None:
           # If any
# single forecast fails, return None for all. Error is logged/displayed in
# helper.
           return None, None, None, None
 
       forecast_results[name] = forecast_series
       mape_metrics[name] = mape
       model_summaries[name] = summary
 
   # 2. Combine
# results into two DataFrames (Historical and Forecast)
    
   # Create the
# unified historical DataFrame
   df_combined_history = pd.DataFrame({
       'Copra_Production (MT)': ts_production,
       'Farmgate Price (PHP/kg)': ts_farmgate,
       'Millgate Price (PHP/kg)': ts_millgate,
       'Type': 'Historical'
   })
    
   # Create the
# unified forecast DataFrame
   future_index = forecast_results['Copra_Production (MT)'].index
   df_combined_forecast = pd.DataFrame({
       'Copra_Production (MT)': forecast_results['Copra_Production (MT)'],
       'Farmgate Price (PHP/kg)': forecast_results['Farmgate Price (PHP/kg)'],
       'Millgate Price (PHP/kg)': forecast_results['Millgate Price (PHP/kg)'],
       'Type': 'Forecast'
   },
index=future_index)
    
   # Combine the two
# resulting dataframes for single display in plots
   df_combined_plot = pd.concat([df_combined_history, df_combined_forecast])
 
 
   return df_combined_plot, df_combined_forecast, mape_metrics, model_summaries
 
 
# --- 4. Page Functions ---
 
def main_page():
   """Displays the single-barangay data editor,
visualization, and ARIMA forecast."""
    
   # st.title string fixed in previous step
   st.title(":coconut: Barangay Production Analysis & Forecasting")
   st.markdown("---")
    
   # Use data from
# session state
   df_current = st.session_state['df_data']
    
   # Get unique
# barangays for selection
   barangays = df_current['Barangay'].unique()
    
   # Sidebar for
# Filtering
   st.sidebar.header("Barangay Selection")
   selected_barangay = st.sidebar.selectbox(
       # FIX: Combined split string into one line
       "Select Barangay for Analysis:", 
       options=barangays,
       key='barangay_select'
    )
    
   # --- A. Data
# Viewer and Editor ---
   st.header(f"1.
Raw Data Viewer & Editor for {selected_barangay}")
   st.info("You
# can directly edit the values below or use the 'Add New Data Point' section to
# append a row.")
 
   # Filter data for
# the selected barangay to display in the editor
   df_barangay_editable = df_current[df_current['Barangay'] ==
selected_barangay].sort_values(by='Period', ascending=True).copy()
 
   # Use
# st.data_editor for interactive editing/deleting of the filtered data
   edited_df = st.data_editor(
       df_barangay_editable,
       column_config={
           "Period": st.column_config.DatetimeColumn("Period",
format="YYYY-MM-DD", disabled=True),
           "Barangay": st.column_config.TextColumn("Barangay",
disabled=True),
           # Added Area (hectares) to allow editing/display
           "Area (hectares)": st.column_config.NumberColumn("Area (hectares)",
min_value=0, format="%d")
        },
       key='data_editor',
       hide_index=True,
       num_rows="dynamic"
    )
 
   # Convert
# edited_df back to the main DataFrame
   if edited_df.shape[0] != df_barangay_editable.shape[0] or not
edited_df.equals(df_barangay_editable):
       # 1. Remove
# old data for the selected barangay from the session state
       df_other_barangays = df_current[df_current['Barangay'] !=
selected_barangay]
        
       # 2. Add the
# newly edited (or deleted/modified) data
       st.session_state['df_data'] = pd.concat([df_other_barangays, edited_df],
ignore_index=True)
       # Rerun is
# usually needed only if the data structure changes, but we rerun later after
# adding data.
        
   # Filter the
# final, processed data for the current barangay
   df_barangay_final = st.session_state['df_data'][st.session_state['df_data']['Barangay'] ==
selected_barangay].copy()
    
   # Convert the
# final DataFrame back to a time series for modeling and visualization
   df_barangay_final['Period'] = pd.to_datetime(df_barangay_final['Period'])
   df_barangay_final = df_barangay_final.set_index('Period').sort_index()
    
   ts_production = df_barangay_final['Copra_Production (MT)']
   ts_farmgate = df_barangay_final['Farmgate Price (PHP/kg)']
   ts_millgate = df_barangay_final['Millgate Price (PHP/kg)']
   last_historical_date = ts_production.index.max() if not ts_production.empty else None
 
   # Get the latest Area value for display
   latest_area = df_barangay_final['Area (hectares)'].iloc[-1] if not df_barangay_final.empty else 0
    
   # --- B. Add New
# Data Point Form ---
   st.header(f"1.5. Add New Data Point")
   with st.expander("Click here to add a new data row"):
       with st.form("add_data_form", clear_on_submit=True):
            
           # Use 6 columns now to fit Area
           col_b, col_p, col_c, col_f, col_m, col_a = st.columns(6)
            
           # Default
# the barangay to the currently selected one
           current_barangay_index = list(barangays).index(selected_barangay) if
selected_barangay in barangays else 0
            
           new_barangay = col_b.selectbox("Barangay", options=barangays,
index=current_barangay_index)
           # Suggest
# the next period's date
           suggested_date = last_historical_date + DateOffset(months=3) if
last_historical_date is not None else pd.to_datetime('2025-10-01')
           new_period = col_p.date_input("Period (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)",
value=suggested_date)
 
           new_copra = col_c.number_input("Copra Production (MT)", min_value=0.0,
format="%.2f")
           new_farmgate = col_f.number_input("Farmgate Price (PHP/kg)",
min_value=0.0, format="%.2f")
           new_millgate = col_m.number_input("Millgate Price (PHP/kg)",
min_value=0.0, format="%.2f")
           # New input for Area
           new_area = col_a.number_input("Area (hectares)", min_value=0,
format="%d")
 
           submitted = st.form_submit_button("Add Data Point and Rerun Analysis")
 
           if submitted:
               if new_period is None:
                   st.error("Please select a Period.")
               else:
                   new_period_dt = pd.to_datetime(new_period)
                   new_data = {
                       'Barangay': new_barangay,
                       'Year': new_period_dt.year,
                       'Quarter': f'Q{(new_period_dt.month - 1) // 3 + 1}',
                       'Period': new_period_dt,
                       'Copra_Production (MT)': new_copra,
                       'Farmgate Price (PHP/kg)':
new_farmgate,
                       'Millgate Price (PHP/kg)': new_millgate,
                       # Added Area to new data point
                       'Area (hectares)': new_area 
                    }
                    
                   #
# Create a temporary DataFrame for the new row
                   new_row_df = pd.DataFrame([new_data])
 
                   #
# Append to session state
                   st.session_state['df_data'] = pd.concat([st.session_state['df_data'],
new_row_df], ignore_index=True)
                   st.success(f"New data point added for **{new_barangay}** on
**{new_period_dt.strftime('%Y-%m-%d')}**. Rerunning app...")
                   st.rerun() # Rerun to update plots and forecasts
 
   # --- C.
# Historical Trend Analysis & Visualization (Production + Prices) ---
   st.header("2.
Historical Trends (Production & Prices)")
   st.subheader(f"Historical Data for {selected_barangay}")
 
   # Display Area as a metric
   st.metric(label="Latest Recorded Area (Hectares)", value=f"{latest_area:,.0f} ha", delta_color="off")
 
   col1, col2 = st.columns(2)
 
   with col1:
       st.caption("Copra Production (Metric Tons)")
       # Production
# Line Plot
       fig_prod, ax_prod = plt.subplots(figsize=(10, 5))
       ts_production.plot(ax=ax_prod, marker='o', linestyle='-',
color='#0077B6', label='Production (MT)')
       ax_prod.set_title(f'Copra Production Trend')
       ax_prod.set_xlabel('Time (Quarterly)')
       ax_prod.set_ylabel('Production (MT)')
       ax_prod.grid(axis='y', linestyle='--')
       ax_prod.legend(loc='upper left')
       st.pyplot(fig_prod)
        
 
   with col2:
       st.caption("Farmgate and Millgate Prices (PHP/kg)")
       # Price Line
# Plot
       fig_price, ax_price = plt.subplots(figsize=(10, 5))
       ts_farmgate.plot(ax=ax_price, marker='s', linestyle='-',
color='#48A9A6', label='Farmgate Price')
       ts_millgate.plot(ax=ax_price, marker='^', linestyle='-',
color='#F4A261', label='Millgate Price')
       ax_price.set_title(f'Copra Price Trends')
       ax_price.set_xlabel('Time (Quarterly)')
       ax_price.set_ylabel('Price (PHP/kg)')
       ax_price.grid(axis='y', linestyle='--')
       ax_price.legend(loc='upper left')
       st.pyplot(fig_price)
 
   # --- D.
# Forecasting ---
   st.header("3.
ARIMA Forecasting (2026 - 2035)")
   if last_historical_date is not None:
       st.caption(f"Forecasting Copra Production and Prices starting from
Q1 of the next period after {last_historical_date.strftime('%Y-%m-%d')}.")
   else:
       st.warning("No historical data available to run the
forecast.")
       return
 
   # Perform the
# forecast pipeline for all three metrics
   # Need to pass
# copies because pandas Series might not be hashable/cacheable if modified in
# place
   df_combined_plot, df_combined_forecast, mape_metrics, model_summaries = arima_forecast(
       ts_production.copy(),
       ts_farmgate.copy(), 
       ts_millgate.copy(), 
       2035,
       last_historical_date
    )
 
   if df_combined_plot is not None:
        
       # --- D1.
# Forecast Visualization (Separate plots for Production and Prices) ---
       st.subheader("Forecast Visualization (Historical +
Predicted)")
        
       col_viz_1, col_viz_2 = st.columns(2)
        
       # Plot 1:
# Production Forecast
       with col_viz_1:
           st.caption("Copra Production Forecast (MT)")
           fig_f_prod, ax_f_prod = plt.subplots(figsize=(10, 5))
            
           # Plot
# Historical Production
           df_combined_plot[df_combined_plot['Type'] ==
'Historical']['Copra_Production (MT)'].plot(
               ax=ax_f_prod, label='Historical Production', color='#1E88E5',
linestyle='-', marker='.'
            )
            
           # Plot
# Forecast Production
           df_combined_plot[df_combined_plot['Type'] ==
'Forecast']['Copra_Production (MT)'].plot(
               ax=ax_f_prod, label='ARIMA Forecast', color='#FF7043', linestyle='--',
marker='.'
            )
            
           ax_f_prod.set_title(f'Copra Production Forecast for
{selected_barangay}')
           ax_f_prod.set_xlabel('Period')
           ax_f_prod.set_ylabel('Copra Production (MT)')
           ax_f_prod.legend()
           ax_f_prod.grid(axis='y', linestyle=':')
           # Draw a
# line at the last historical point
           ax_f_prod.axvline(x=last_historical_date, color='grey', linestyle=':',
linewidth=2, label='Forecast Start')
           st.pyplot(fig_f_prod)
            
       # Plot 2:
# Price Forecast (Farmgate & Millgate)
       with col_viz_2:
           st.caption("Price Forecast (Farmgate & Millgate Price)")
           fig_f_price, ax_f_price = plt.subplots(figsize=(10, 5))
 
           # Plot
# Historical Prices
           df_hist = df_combined_plot[df_combined_plot['Type'] == 'Historical']
           df_hist['Farmgate Price (PHP/kg)'].plot(ax=ax_f_price, label='Historical
Farmgate', color='#00A896', linestyle='-', marker='s')
           df_hist['Millgate Price (PHP/kg)'].plot(ax=ax_f_price, label='Historical
Millgate', color='#F4B400', linestyle='-', marker='^')
 
           # Plot
# Forecast Prices
           df_fore = df_combined_plot[df_combined_plot['Type'] == 'Forecast']
           df_fore['Farmgate Price (PHP/kg)'].plot(ax=ax_f_price, label='Forecast
Farmgate', color='#00A896', linestyle='--', alpha=0.7)
            df_fore['Millgate Price
# (PHP/kg)'].plot(ax=ax_f_price, label='Forecast Millgate', color='#F4B400',
linestyle='--', alpha=0.7)
 
           ax_f_price.set_title(f'Price Forecast for {selected_barangay}')
           ax_f_price.set_xlabel('Period')
           ax_f_price.set_ylabel('Price
# (PHP/kg)')
           ax_f_price.legend(loc='upper left')
           ax_f_price.grid(axis='y', linestyle=':')
           # Draw a
# line at the last historical point
           ax_f_price.axvline(x=last_historical_date, color='grey', linestyle=':',
linewidth=2, label='Forecast Start')
           st.pyplot(fig_f_price)
 
       # --- D2.
# Forecast Metrics & Table ---
       st.subheader("Forecast Metrics & Data")
        
       mape_col1, mape_col2, mape_col3 = st.columns(3)
        
       with mape_col1:
           st.metric(
               label="Production MAPE", 
               value=mape_metrics['Copra_Production (MT)'],
               help="MAPE is calculated by backtesting the model on the last 4
known historical quarters to estimate predictive accuracy for Production."
            )
            
       with mape_col2:
           st.metric(
               label="Farmgate Price MAPE", 
               value=mape_metrics['Farmgate Price (PHP/kg)'],
               help="MAPE is calculated by backtesting the model on the last 4
known historical quarters to estimate predictive accuracy for Farmgate
Price."
            )
 
       with mape_col3:
           st.metric(
               label="Millgate Price MAPE", 
               value=mape_metrics['Millgate Price (PHP/kg)'],
               help="MAPE is calculated by backtesting the model on the last 4
known historical quarters to estimate predictive accuracy for Millgate
Price."
            )
 
 
       st.markdown("**Forecasted Data Table
(Production and Prices)**")
       df_table = df_combined_forecast.copy()
       df_table.index.name = 'Forecast Period'
       df_table['Year'] = df_table.index.year
       df_table['Quarter'] = df_table.index.quarter.map({1: 'Q1', 2: 'Q2', 3:
'Q3', 4: 'Q4'})
        
       # Round
# numerical columns for display
       for col in ['Copra_Production (MT)', 'Farmgate Price (PHP/kg)', 'Millgate Price
(PHP/kg)']:
            df_table[col] = df_table[col].round(2)
        
       # Final table
# display
       st.dataframe(
           df_table[['Year', 'Quarter', 'Copra_Production (MT)', 'Farmgate Price
(PHP/kg)', 'Millgate Price (PHP/kg)']],
           height=300
        )
 
       # --- D3.
# Model Diagnostics (Optional) ---
       with st.expander("View All ARIMA Model Summaries"):
           st.subheader("Copra Production Model Summary (ARIMA(1, 1,
0))")
           st.code(model_summaries['Copra_Production (MT)'])
            
           st.subheader("Farmgate Price Model Summary (ARIMA(1, 1, 0))")
           st.code(model_summaries['Farmgate Price (PHP/kg)'])
            
           st.subheader("Millgate Price Model Summary (ARIMA(1, 1, 0))")
           st.code(model_summaries['Millgate Price (PHP/kg)'])
            
           st.caption("Note: The model used is a simple ARIMA(1, 1, 0) for
demonstration purposes. Results may vary.")
 
   else:
       # If
# model_summaries is None, an error occurred in the pipeline
        st.error("Forecasting could not be
# completed due to insufficient data or model errors. Check console for
# details.")
 
   st.markdown("---")
 
def comparison_page():
   """Displays comparative visualizations for all barangays,
using session state data."""
    
   st.title(":chart_with_upwards_trend: All Barangays
Comparison")
   st.markdown("---")
    
   df_current = st.session_state['df_data']
 
   st.header("1.
Production Comparison (Metric Tons)")
    
   # Group and pivot
# data for plotting all series
   df_pivot_prod = df_current.pivot_table(
       index='Period', 
       columns='Barangay', 
       values='Copra_Production (MT)'
    )
    
   # Plot Production
# Comparison
   fig_prod, ax_prod = plt.subplots(figsize=(12, 6))
   df_pivot_prod.plot(ax=ax_prod, marker='.', linestyle='-')
   ax_prod.set_title('Copra Production (MT) Comparison Across All
Barangays')
   ax_prod.set_xlabel('Period')
   ax_prod.set_ylabel('Copra Production (MT)')
   ax_prod.legend(title='Barangay', bbox_to_anchor=(1.05, 1), loc='upper
left')
   ax_prod.grid(axis='y', linestyle=':')
   plt.tight_layout()
   st.pyplot(fig_prod)
    
   st.markdown("---")
    
   st.header("2.
Price Comparison (Farmgate & Millgate)")
    
   col1, col2 = st.columns(2)
    
   # Plot Farmgate
# Price Comparison
   with col1:
       df_pivot_farm = df_current.pivot_table(
           index='Period', 
           columns='Barangay', 
           values='Farmgate Price (PHP/kg)'
        )
       fig_farm, ax_farm = plt.subplots(figsize=(10, 5))
       df_pivot_farm.plot(ax=ax_farm, marker='.', linestyle='-')
       ax_farm.set_title('Farmgate Price (PHP/kg) Comparison')
       ax_farm.set_xlabel('Period')
       ax_farm.set_ylabel('Price (PHP/kg)')
       ax_farm.legend(title='Barangay', fontsize=8, loc='upper left')
       ax_farm.grid(axis='y', linestyle=':')
       plt.tight_layout()
       st.pyplot(fig_farm)
 
   # Plot Millgate
# Price Comparison
   with col2:
       df_pivot_mill = df_current.pivot_table(
           index='Period', 
           columns='Barangay', 
           values='Millgate Price (PHP/kg)'
        )
       fig_mill, ax_mill = plt.subplots(figsize=(10, 5))
       df_pivot_mill.plot(ax=ax_mill, marker='.', linestyle='-')
       ax_mill.set_title('Millgate Price (PHP/kg) Comparison')
       ax_mill.set_xlabel('Period')
       ax_mill.set_ylabel('Price (PHP/kg)')
       ax_mill.legend(title='Barangay', fontsize=8, loc='upper left')
       ax_mill.grid(axis='y', linestyle=':')
       plt.tight_layout()
       st.pyplot(fig_mill)
 
   st.markdown("---")
 
   # New section for Area Comparison
   st.header("3. Planted Area Comparison (Hectares)")
 
   # Group and pivot data for plotting all series
   df_pivot_area = df_current.pivot_table(
       index='Period', 
       columns='Barangay', 
       values='Area (hectares)'
    )
 
   # Plot Area Comparison
   fig_area, ax_area = plt.subplots(figsize=(12, 6))
   df_pivot_area.plot(ax=ax_area, marker='.', linestyle='-')
   ax_area.set_title('Planted Area (Hectares) Comparison Across All Barangays')
   ax_area.set_xlabel('Period')
   ax_area.set_ylabel('Area (Hectares)')
   ax_area.legend(title='Barangay', bbox_to_anchor=(1.05, 1), loc='upper left')
   ax_area.grid(axis='y', linestyle=':')
   plt.tight_layout()
   st.pyplot(fig_area)
 
 
# --- 5. Main App Navigation ---
 
def run_app():
   """Main function to run the Streamlit app with
navigation."""
    
   # Setup Streamlit
# page configuration
   st.set_page_config(layout="wide", page_title="Copra
Production & Price Dashboard")
    
   # Initialize data
# into session state
   initialize_session_data()
    
   # Sidebar
# Navigation
   st.sidebar.title("Navigation")
   page = st.sidebar.radio(
       "Select a Page",
       ("Barangay Forecast & Analysis", "All Barangays
Comparison")
    )
    
   # Display the
# selected page
   if page == "Barangay Forecast & Analysis":
       main_page()
   elif page == "All Barangays Comparison":
        comparison_page()
 
if __name__ == "__main__":
   run_app()
