VERSION = '0.1.0'

DATASET = "loan_final313.csv"

PATH_TO_DATASET = 'data/'

ACCEPTABLE_COLUMNS = ['id', 'year', 'issue_d', 'final_d', 'emp_length_int', 'home_ownership',
	   'home_ownership_cat', 'income_category', 'annual_inc', 'income_cat',
	   'loan_amount', 'term', 'term_cat', 'application_type',
	   'application_type_cat', 'purpose', 'purpose_cat', 'interest_payments',
	   'interest_payment_cat', 'loan_condition', 'loan_condition_cat',
	   'interest_rate', 'grade', 'grade_cat', 'dti', 'total_pymnt',
	   'total_rec_prncp', 'recoveries', 'installment', 'region']

CATEGORICAL_COLUMNS = ['issue_d', 'home_ownership', 'income_category', 'term', 'application_type', 'purpose', 'interest_payments', 'grade', 'region']

NUMERICAL_COLUMNS = ['id', 'year', 'final_d', 'emp_length_int', 'home_ownership_cat', 'annual_inc', 'income_cat', 'loan_amount', 'term_cat', 'application_type_cat', 'purpose_cat', 'interest_payment_cat', 'loan_condition', 'loan_condition_cat', 'interest_rate', 'grade_cat', 'dti', 'total_pymnt', 'total_rec_prncp', 'recoveries', 'installment']

DROP_THIS = ['id']

UNNECESSARY_COLUMNS = ['year','issue_d','home_ownership_cat','income_cat','term_cat','application_type_cat','purpose_cat','interest_payment_cat','loan_condition_cat','grade_cat']

TARGET_COLUMN = 'loan_condition'