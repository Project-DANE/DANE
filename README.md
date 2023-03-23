# DANE

# Project Goals

The goal of this project is to create a secondary screening model that will detect and flag any applicant that is likely to cause some form of loss to the company either through damages, delinquency, uncollectalbe fees, or other similar charges. We partnered with CWS Capital Partners to aquire data for this project. The focus for our model will be to ensure as few 'bad residents' as possible make it through our screen.

# Project Description

To acomplish our goals first we will aquire the data from CWS containing information on just over 5000 residents across 97 different properties from the year 2018 until this year. Then we will begin selecting a feature set from this data to use in our exploration. After selecting the best features and analyzing the data in our explore phase, we will then prepare the data for modeling. Since there is a heavy class imbalence we will use imblearn to change the sampling ratio to better train our models. After finding the best model and running it through the test set, we will then conclude with takeaways from the model and the data as well as some recommendations moving forward for CWS and what we believe to be the best application for our model.

# Initial Questions
1. What is the average risk score?
2. Is risk score an indicator of loss?
3. Is monthly/yearly income an indicator of loss?
4. Is requirement of a guarantor a good driver?
5. Is there a relationship between rent and damages?
6. Are certain age groups more likely to cause damage?
7. Are short term renters more likely to cause damage?
8. Do certain property has more damage?
9. What is the most common damage code?


# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from SQL Server of CWS Capital Partners, a real estate investment management
### Prepare Data

 - Clean and prepare the data 
 - Split data into train, validate, and test.
 
### Explore Data

- Create visuals on data 
- Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.

### Develop Model

 - Isolate target variable
 - Create, Fit, Predict on train.
 - Evaluate models on train and validate datasets.
 - Select the best model
 - Run the best model on test data to make predictions
 
### Delivery  
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for screening an applicant. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      bad_resident    | 1 for resident that caused loss to company, 0 for otherwise |

| Feature  | Definition |
| ------------- | ------------- |
| id | Tenant Identification Number |
| total_charges | Total charges resident has incurred living at apartment   |
| amount_paid | Amount tenant has paid towards total charges |
| open |  0 is transaction total charge is closed -1 is transaction total charge open|
| charge_code | Code of a charge  |
| description | Description of a charge with date|
| prop_id | Related to property name and location  |
| charge_name | Name of a charge  |
| sStatus | Status of a tenant (current or past) |
| rent | Rent per month |
| term | Lease contract in months |  
| monthly_income | Monthly income of a tenant |
| GuarantorRequired | If a tenant requires a guarrantor |
| total_income | Yearly income of a tenant |
| Recommendation | Acceptance status of applicant |
| age | Age of a tenant |
| risk_score | Predetermined score of how risky an applicant is at the time of applying |
| reason | Reason that drove decision to recommedation |



## Steps to Reproduce

- Contact CWS Capital to request data
- Clone this repo
- Put the data in a file containing the cloned repo.
- Run notebook.
(An important note: We do not expect this project to be possible to reproduce since the data was aquired from a private company)

## Conclusions

**TBD **


 
**Best Model's performance**
- imblearn's Easy Ensemble model performed the best at 63% recall of our 'positive outcome' (being classified as a 'bad resident') on our validate set and 55% on our test set.
- Since there was no baseline for this project we have beat the baseline of 0%.
- We also managed to maintain over 50% recall at an acceptable loss of precision

## Recommendations
- We would recommend to use our model as a way to screen applicants to minimize damage or loss to the company. It's important to note that this screening would be a secondary screening process and not the primary one. We made our model in an efficient way to cooperate with the intital screening process.
- We would also recommend looking into Georgia's properties and findning why this state tends to have more damage/loss charges.

## Next Steps

- Pull more data from the SQL server to find more correlations and patterns to improve our model
- Understand more of the primary application process
