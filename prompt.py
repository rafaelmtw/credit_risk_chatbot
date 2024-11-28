# Age (numeric)
# Sex (text: male, female)
# Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# Housing (text: own, rent, or free)
# Saving accounts (text - little, moderate, quite rich, rich)
# Checking account (numeric, in DM - Deutsch Mark)
# Credit amount (numeric, in DM)
# Duration (numeric, in month)
# Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

age_prompt = """Evaluate the following user input to determine the age. If the age can be inferred, output it as an integer in the format {age: X}. If it cannot be determined, respond with {age: 'irrelevant'}: '{user_input}'"""

sex_prompt = """Evaluate the following user input to determine the sex (male or female). If the sex can be inferred, output it as {sex: 'male'} or {sex: 'female'}. If it cannot be determined, respond with {sex: 'irrelevant'}: '{user_input}'"""

job_prompt = """Evaluate the following user input to determine the job status (0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled). If the job can be inferred, output it as {job: X}. If it cannot be determined, respond with {job: 'irrelevant'}: '{user_input}'"""

housing_prompt = """Evaluate the following user input to determine the housing situation (own, rent, or free). If the housing situation can be inferred, output it as {housing: 'own'}, {housing: 'rent'}, or {housing: 'free'}. If it cannot be determined, respond with {housing: 'irrelevant'}: '{user_input}'"""

saving_accounts_prompt = """Evaluate the following user input to determine the savings accounts status = (little, moderate, quite rich, rich). If the savings status can be inferred, output it as {saving_accounts: 'status'}. If it cannot be determined, respond with {saving_accounts: 'irrelevant'}: '{user_input}'"""

checking_account_prompt = """Evaluate the following user input to determine the checking account balance in DM. If the balance can be inferred, output it as {checking_account: X}, let X is integer. If it cannot be determined, respond with {checking_account: 'irrelevant'}: '{user_input}'"""

credit_amount_prompt = """Evaluate the following user input to determine the credit amount they are seeking in DM. If the amount can be inferred, output it as {credit_amount: X}, let X is integer. If it cannot be determined, respond with {credit_amount: 'irrelevant'}: '{user_input}'"""

duration_prompt = """Evaluate the following user input to determine the credit duration in months. If the duration can be inferred, output it as {duration: X}, let X is number of month. If it cannot be determined, respond with {duration: 'irrelevant'}: '{user_input}'"""

purpose_prompt = """Evaluate the following user input to determine the purpose of the credit, purpose = (car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others), choose one out of the tuples. If the purpose can be inferred, output it as {purpose: 'type'}. If it cannot be determined, respond with {purpose: 'irrelevant'}: '{user_input}'"""