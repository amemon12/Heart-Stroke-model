import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np



# Function to remove outliers based on Z-score for individual columns
def remove_outliers_zscore_column(df, column, threshold=4):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    mask = z_scores < threshold
    return df[mask]

# Read the dataset
df = pd.read_csv("C:\\Users\\admin\\Downloads\\Heart Stroke dataset.csv")

# Remove outliers
columns_to_remove_outliers = ['SPP', 'DPP', 'Cholesterol Levels(HDL)', 'Stress Levels', 'Cholesterol Levels(LDL)']
for column in columns_to_remove_outliers:
    df = remove_outliers_zscore_column(df, column)

# Label Encoding
le = LabelEncoder()
categorical_columns = ['Sex', 'Smoking Status', 'Stroke History', 'Diagnosis', 'Dietary Habits']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    print("Encoded Classes",col,le.classes_)

# Apply transformations
df['DPP'] = df['DPP'].apply(lambda x: 2 if x > 119 else (1 if x > 39 else 0))
df['SPP'] = df['SPP'].apply(lambda x: 2 if x > 79 else (1 if x > 39 else 0))
df['Cholesterol Levels(HDL)'] = df['Cholesterol Levels(HDL)'].apply(lambda x: 2 if x > 59 else (1 if  x > 39 else 0))
df['Cholesterol Levels(LDL)'] = df['Cholesterol Levels(LDL)'].apply(lambda x: 2 if x > 129 else (1 if x > 39 else 0))
df['Stress Levels'] = df['Stress Levels'].apply(lambda x: 2 if x > 4 else (1 if x >= 4 else 0))

# Splitting data into features and target
x = df.iloc[:, 1:12]
y = df['Diagnosis']

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=50)

# Training the model
svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
svm.fit(x, y)


# Evaluating the model
y_pred = svm.predict(x_test)
print("Let's Compare with in % Accuracy")
print("Accuracy score ", accuracy_score(y_test, y_pred))
print("Confusion Matrix", confusion_matrix(y_test, y_pred))

#Predictions
P_sex = input("Enter sex (male/female): ").lower()  # male/female

if P_sex == 'male':
    Pe_Sex = 0
else:
    Pe_Sex = 1

P_Avg_glu = float(input("Enter Avg Glucose levels: "))
P_Bmi = float(input("Enter BMI: "))
P_smoke_status = input("Enter Smoking status (Currently Smokes/Formerly Smoked/Non-smoker): ").lower()
# Currently Smokes/Formerly Smoked/Non-smoker

if P_smoke_status == 'currently smokes':
    Pe_Smoke_Status = 0
elif P_smoke_status == 'formerly smoked':
    Pe_Smoke_Status = 1
else:
    Pe_Smoke_Status = 2  # Non-smoker

P_Stroke_History = input("If any previous stroke (yes/no): ").lower()  # yes/no

if P_Stroke_History == 'yes':
    Pe_Stroke_History = 1
else:
    Pe_Stroke_History = 0

P_Dietary_Habits = input("What type of Diet (Gluten-Free/Keto/Non-Vegetarian/Paleo/Pescatarian/Vegan/Vegetarian): ").lower()
# Gluten-Free/Keto/Non-Vegetarian/Paleo/Pescatarian/Vegan/Vegetarian

diets = ['gluten-free', 'keto', 'non-vegetarian', 'paleo', 'pescatarian', 'vegan', 'vegetarian']
Pe_Dietary_Habits = diets.index(P_Dietary_Habits)

P_stress_levels = float(input("Stress Level Scores: "))
P_Cholesterol_HDL = int(input("Enter Cholesterol HDL levels: "))
P_Cholesterol_LDL = int(input("Enter Cholesterol LDL levels: "))
P_BP_DPP = int(input("Enter Diastolic Blood pressure (DPP/SPP): "))
P_BP_SPP = int(input("Enter Systolic Blood pressure (DPP/SPP): "))

# Encoding based on specified criteria
Pe_Stress_Levels = 2 if P_stress_levels > 4 else (1 if P_stress_levels >= 4 else 0)
Pe_Cholesterol_HDL = 2 if P_Cholesterol_HDL > 59 else (1 if P_Cholesterol_HDL > 39 else 0)
Pe_Cholesterol_LDL = 2 if P_Cholesterol_LDL > 129 else (1 if P_Cholesterol_LDL > 39 else 0)
Pe_BP_DPP = 2 if P_BP_DPP > 119 else (1 if P_BP_DPP > 39 else 0)
Pe_BP_SPP = 2 if P_BP_SPP > 79 else (1 if P_BP_SPP > 39 else 0)

new_data = [[Pe_Sex, P_Avg_glu, P_Bmi, Pe_Smoke_Status, Pe_Stroke_History, Pe_Dietary_Habits, Pe_Stress_Levels, Pe_Cholesterol_HDL, Pe_Cholesterol_LDL, Pe_BP_DPP, Pe_BP_SPP]]

predicted_class = svm.predict(new_data)

if predicted_class==1:
    print(" High Chances of Stroke")
else:
    print(" Low chances of stroke")


