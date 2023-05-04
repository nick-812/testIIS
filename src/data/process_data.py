import pandas as pd

df0 = pd.read_csv("data/raw/students_grades.csv", sep=",", header=0)
df1 = pd.read_excel("data/raw/students_school.xlsx")
df2 = pd.read_csv("data/raw/students.txt", sep=";", header=0)

df = pd.merge(df0, df1, on="student_id")
df = pd.merge(df, df2, on="student_id")

df['final_grade'] = (df['G1'] + df['G2'] + df['G3'])/3

df = df.drop(['G1'], axis=1)
df = df.drop(['G2'], axis=1)
df = df.drop(['G3'], axis=1)

df.to_csv('data/processed/current_data.csv')

print(df)
