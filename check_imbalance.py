import pandas as pd

def check_imbalance(csv_path, target_col):
    data = pd.read_csv(csv_path)
    y = data[target_col]
    print('Class counts:')
    print(y.value_counts())
    print('\nClass proportions:')
    print(y.value_counts(normalize=True))

if __name__ == "__main__":
    # Update these as needed
    csv_path = 'loans.csv'
    target_col = 'Loan_Status'
    check_imbalance(csv_path, target_col)
