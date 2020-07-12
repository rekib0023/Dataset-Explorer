import os
import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


def main():
	""" Common ML Dataset Explorer """
	st.title("Common ML Dataset Explorer")
	st.subheader("Simple Data Science Explorer with Streamlit")

	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:36px;">StreamLit is Awesome</p.</div>
	"""

	st.markdown(html_temp, unsafe_allow_html=True)

	def file_selector(folder_path='.\\datasets'):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox('Select a file', filenames)
		return os.path.join(folder_path,selected_filename)

	filename = file_selector()
	file = filename.split("\\")[-1]
	st.info(f"You selected {file}")


	# Read Data
	df = pd.read_csv(filename)

	# Show Data
	if st.checkbox("Show Dataset"):
		number = st.number_input("Number of rows to view",5)
		st.dataframe(df.head(number))

	# Show Columns
	# if st.button("Column Names"):
	# 	st.write(df.columns)

	# Show Shape
	if st.button("Shape"):
		row = df.shape[0]
		column = df.shape[1]
		st.write("Number of rows:", row)
		st.write("Number of columns:", column)
	
	# Missing Values
	if st.button("Missing Values"):
		missing_percentage = df.isna().mean()
		missing_sum = df.isna().sum()
		missing_df = pd.DataFrame([missing_percentage,missing_sum],index=['Percentage', 'Total'],columns=df.columns)
		st.dataframe(missing_df.T)

	# Datatypes
	if st.button("Datatypes"):
		st.write(df.dtypes)

	# Summary
	if st.checkbox("Get a Statistical Summary"):
		st.write(df.describe().T)

	# Select Columns
	if st.button("Column Names"):
		st.write(df.columns)


	st.header("Visualizations")

if __name__ == '__main__':
	main()