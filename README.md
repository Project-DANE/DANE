# DANE

# Project Goals

 - TBD

# Project Description

TBD.

# Initial Questions

 TBD


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
| monthly_inc | Monthly income of a tenant |
| GuarantorRequired | If a tenant requires a guarrantor |
| total_inc | Yearly income of a tenant |
| Recommendation | Acceptance status of applicant |
| age | Age of a tenant |
| risk_score | Predetermined score of how risky an applicant is at the time of applying |
| reason | Reason that drove decision to recommedation |



## Steps to Reproduce

- Clone this repo
- Put the data in a file containing the cloned repo.
- Run notebook.

## Conclusions

**TBD**


 
**Best Model's performance**
**-TBD**

**- TBD**

## Recommendations
- TBD

## Next Steps

- TBD
