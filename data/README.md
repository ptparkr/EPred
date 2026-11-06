# UCI Appliances Energy Prediction Dataset

## Dataset Source & Attribution
- **Dataset Name**: Appliances Energy Prediction
- **Repository**: UCI Machine Learning Repository
- **DOI / URL**: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- **Authors**: Luis M. Candanedo, Véronique Feldheim, Dominique Deramaix
- **Citation**: 
  > Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81-97.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset Description
The dataset contains 19,735 observations logged at 10-minute intervals from January 11, 2016 17:00:00 to May 27, 2016 18:00:00 (approximately 4.5 months) in a low-energy house in Stambruges, Belgium.

### Variables:
1. `date`: Timestamp in year-month-day hour:minute:second format (UTC+1)
2. `Appliances`: Energy use in Wh of appliances in the house (TARGET basis)
3. `lights`: Energy use of light fixtures in the house in Wh
4. `T1`: Temperature in kitchen area, in Celsius
5. `RH_1`: Humidity in kitchen area, in %
6. `T2`: Temperature in living room area, in Celsius
7. `RH_2`: Humidity in living room area, in %
8. `T3`: Temperature in laundry room area, in Celsius
9. `RH_3`: Humidity in laundry room area, in %
10. `T4`: Temperature in office room, in Celsius
11. `RH_4`: Humidity in office room, in %
12. `T5`: Temperature in bathroom, in Celsius
13. `RH_5`: Humidity in bathroom, in %
14. `T6`: Temperature outside the building (north side), in Celsius
15. `RH_6`: Humidity outside the building (north side), in %
16. `T7`: Temperature in ironing room, in Celsius
17. `RH_7`: Humidity in ironing room, in %
18. `T8`: Temperature in teenager room 2, in Celsius
19. `RH_8`: Humidity in teenager room 2, in %
20. `T9`: Temperature in parents' room, in Celsius
21. `RH_9`: Humidity in parents' room, in %
22. `T_out`: Temperature outside (from Chièvres weather station), in Celsius
23. `Press_mm_hg`: Pressure (from Chièvres weather station), in mm Hg
24. `RH_out`: Humidity outside (from Chièvres weather station), in %
25. `Windspeed`: Wind speed (from Chièvres weather station), in m/s
26. `Visibility`: Visibility (from Chièvres weather station), in km
27. `Tdewpoint`: Dew point temperature (from Chièvres weather station), in °C
28. `rv1`: Random variable 1, nondimensional (excluded from modeling)
29. `rv2`: Random variable 2, nondimensional (excluded from modeling)

## Important Distinction
- `Appliances` measures the electric energy consumption of household appliances (Wh).
- `lights` measures lighting energy consumption (Wh).
- `Appliances` does NOT represent total building electricity consumption.
