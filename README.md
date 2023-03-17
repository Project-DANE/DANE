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
|      bad_resident    | 1 or 0 |

| Feature  | Definition |
| ------------- | ------------- |
| id | Customer id |
| total_charges | Total charge  |
| amount_paid | Charge |
| open |  Is transaction for a charge open or close?|
| charge_code | Code of a charge  |
| description | Description of a charge |
| prop_id | Property id |
| charge_name | Name of a charge  |
| sStatus | Status of a customer |
| rent | Rent of a month |
| term | Rent period in a month |  
| monthly_inc | Monthly income of a customer |
| GuarantorRequired | Does a customer require a guarrantor? True or False |
| total_inc | Yearly income of a customer |
| Recommendation | Steps to take for an applicant |
| age | Age of a customer |
| risk_score | predetermined score of an applicant associated with a risk|
| reason | Reason that drove decision |



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
