/* Show the number of solar panel installations up to 2015 for this municipality */
SELECT COUNT(*) AS num_installations
FROM installation
WHERE Year_Installed <= '2015-12-31'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth';

/* Show the number of solar panel installations up to 2020 for this municipality */
SELECT COUNT(*) AS num_installations
FROM installation
WHERE Year_Installed <= '2020-12-31'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth';

/* Show the number of residential solar panel installations for this municipality up to 2015 */
SELECT COUNT(*) AS num_residential_installations
FROM installation
WHERE Year_Installed <= '2015-12-31'
  AND Customer_Type = 'Residential'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth';

/* Show the number of residential solar panel installations for this municipality up to 2020 */
SELECT COUNT(*) AS num_residential_installations
FROM installation
WHERE Year_Installed <= '2020-12-31'
  AND Customer_Type = 'Residential'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth';

/* Show the number of commercial solar panel installations for this municipality up to 2015 */
SELECT COUNT(*) AS num_commercial_installations
FROM installation
WHERE Year_Installed <= '2015-12-31'
  AND Customer_Type = 'Commercial'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth';

/* Show the number of commercial solar panel installations for this municipality up to 2020 */
SELECT COUNT(*) AS num_commercial_installations
FROM installation
WHERE Year_Installed <= '2020-12-31'
  AND Customer_Type = 'Commercial'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth';

/* Show the number of other solar panel installations for this municipality up to 2015 */
SELECT COUNT(*) AS num_other_installations
FROM installation
WHERE Year_Installed <= '2015-12-31'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth'
  AND Customer_Type != 'Residential'
  AND Customer_Type != 'Commercial';

/* Show the number of other solar panel installations for this municipality up to 2020 */
SELECT COUNT(*) AS num_other_installations
FROM installation
WHERE Year_Installed <= '2020-12-31'
  AND Municipality = 'Aberdeen township'
  AND County = 'Monmouth'
  AND Customer_Type != 'Residential'
  AND Customer_Type != 'Commercial';

/* Show the total MTCO2e in 2020 for this municipality */
SELECT TOTAL_MTCO2e AS Total_CO2_2020
FROM emission
WHERE Year = '2020'
AND Municipality = 'Aberdeen township'
AND County = 'Monmouth';

/* Show the total MTCO2e in 2015 for this municipality */
SELECT TOTAL_MTCO2e AS Total_CO2_2015
FROM emission
WHERE Year = '2015'
AND Municipality = 'Aberdeen township'
AND County = 'Monmouth';
