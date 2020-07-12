import os
import streamlit as st

import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import norm, skew

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px


def main():
	""" Common ML Dataset Explorer """
	st.title("Common ML Dataset Explorer")
	st.subheader("Now explore your dataset with just a few clicks")

	uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	if uploaded_file is not None:
		df = pd.read_csv(uploaded_file)


		# Show Data
		if st.checkbox("Show Dataset"):
			st.write("Number of rows:", df.shape[0])
			st.write("Number of columns:", df.shape[1])
			number = st.number_input("Number of rows to view",5)
			st.dataframe(df.head(number))

		all_columns = df.columns.sort_values().tolist()
		num_columns = [var for var in all_columns if df[var].dtypes != 'O']
		cat_columns = [var for var in all_columns if var not in num_columns]



		# Datatypes
		if st.button("Datatypes"):
			st.write(df.dtypes)

		# Show Columns
		if st.button("Column Names"):
			st.write("All Features:")
			st.write(df.columns)
			st.write("Numerical Features:")
			st.write(df[num_columns].columns)
			st.write("Categorical Features:")
			st.write(df[cat_columns].columns)



		# Show Columns data
		if st.checkbox("Show Columns Data"):
			cols_to_show = st.multiselect("Select columns to view data", all_columns)
			if not cols_to_show:
				pass
			else:
				st.dataframe(df[cols_to_show])

		
		# Missing Values
		if st.checkbox("Missing Values"):
			missing_percentage = df.isna().mean()
			missing_sum = df.isna().sum()
			missing_df = pd.DataFrame([missing_percentage,missing_sum],index=['Percentage', 'Total'],columns=df.columns)
			st.dataframe(missing_df.T)

		# Summary
		if st.checkbox("Get a Statistical Summary"):
			st.write(df.describe().T)



		st.header("Visualizations")
		st.subheader("Visualize the dataset with interactive plots")
		st.markdown("""<br>""", unsafe_allow_html=True)
		

		if st.button("Show Correlation"):
			corr_matrix = df.corr()

			mask = np.zeros_like(corr_matrix, dtype=np.bool)
			mask[np.triu_indices_from(mask)]= True
			st.write(sns.heatmap(corr_matrix, 
	          	square = True,
	          	linewidths = .5,
	          	cmap = 'coolwarm',
	          	cbar_kws = {'shrink': .4, 
	                    'ticks' : [-1, -.5, 0, 0.5, 1]},
	          	vmin = -1, 
	          	vmax = 1))
			st.pyplot()


		type_of_plot = st.selectbox("Select Type of Plot",["Count Plot", "Bar Plot", "Histogram", "Pie Plot", "Scatter Plot", "Trendline", "Area Plot"])

		# Count Plot
		if type_of_plot == "Count Plot":
			selected_column = st.selectbox("Select Column to Plot", all_columns)
			if st.button("Generate Count Plot"):
				st.write(sns.countplot(x=selected_column, data=df))
				st.pyplot()

		# Bar Plot
		elif type_of_plot == "Bar Plot":
			bar_plot_type = st.radio("Choose a Bar Plot Type",["Bar","Horizontal Bar","Stacked Bar","Group Bar"])
			x = st.selectbox("Select feature for x-axis", all_columns)
			y = st.selectbox("Select feature for y-axis", all_columns)
			color = st.selectbox("Select a column for color", all_columns)
			if x != y:

				# Bar Plot
				if bar_plot_type == "Bar":
					if st.button("Generate Plot"):
						st.write(bar_plot_type)
						grs = df.groupby(x)[y].mean().reset_index()
						fig = px.bar(grs[[x,y]].sort_values(y, ascending=False),
							y=y, x=x, color=x, template='ggplot2')
						st.plotly_chart(fig, use_container_width=True)

				# Horizontal Bar Plot
				elif bar_plot_type == "Horizontal Bar":
					if st.button("Generate Plot"):
						st.write(bar_plot_type)
						grs = df.groupby(x)[y].mean().reset_index()
						fig = px.bar(grs[[x,y]].sort_values(y, ascending=False),
							y=x, x=y, color=x, orientation='h')
						st.plotly_chart(fig, use_container_width=True)

				# Stacked Bar Plot
				elif bar_plot_type == "Stacked Bar":
					if st.button("Generate Plot"):
						st.write(bar_plot_type)
						grgs = df.groupby([x,color])[[y]].mean().reset_index()
						fig = px.bar(grgs, x=x, y=y, color=color, barmode='stack',
						             height=400)
						st.plotly_chart(fig, use_container_width=True)

				# Group Bar Plot
				elif bar_plot_type == "Group Bar":
					if st.button("Generate Plot"):
						grgs = df.groupby([x,color])[[y]].mean().reset_index()
						fig = px.bar(grgs, x=x, y=y, color=color, barmode='group',
						             height=400)
						st.plotly_chart(fig, use_container_width=True)

			else:
				st.error("The values for x and y axes cannot be same")

		# Histogram
		elif type_of_plot == "Histogram":
			var = st.selectbox("Select Column:", num_columns)
			if st.button("Generate Histogram"):
				st.write(type_of_plot)
				df2 = df.copy()
				df2.dropna(inplace=True)
				f,ax = plt.subplots()
				sns.set_style("darkgrid")
				sns.distplot(df2[var], fit=norm)
				plt.legend(['Skewness={:.2f} Kurtosis={:.2f}'.format(
		            df2[var].skew(), 
		            df2[var].kurt())
		        ],
		        loc='best')
				st.pyplot()


		# Pie Plot
		elif type_of_plot == "Pie Plot":
			values = st.selectbox("Select values column:", all_columns)
			labels = st.selectbox("Select labels column:", all_columns)
			if st.button("Generate Pie Plot"):
				st.write(type_of_plot)
				grdsp = df.groupby([labels])[[values]].mean().reset_index()

				fig = px.pie(grdsp,
				             values=values,
				             names=labels)
				fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
				st.plotly_chart(fig, use_container_width=True)


		# Scatter Plot
		elif type_of_plot == "Scatter Plot":
			x = st.selectbox("Select feature for X-axis", all_columns)
			y = st.selectbox("Select feature for y-axis", all_columns)
			if st.button("Generate Scatter Plot"):
				st.write(type_of_plot)
				plt.scatter(df[x], df[y])
				plt.xlabel(x)
				plt.ylabel(y)
				st.pyplot()


		elif type_of_plot == "Trendline":
			line_type = st.radio("Line Type",["Single","Multiple"])
			x = st.selectbox("Select feature for X-axis", all_columns)
			y = st.selectbox("Select feature for y-axis", all_columns)
			if line_type == "Multiple":
				color = st.selectbox("Select feature for color", all_columns)
			if st.button("Generate Trendline"):
				if line_type == "Multiple":
					fig = px.scatter(df, x=x, y=y, color=color, trendline="ols")
				else:
					fig = px.scatter(df, x=x, y=y, trendline="ols")
				st.plotly_chart(fig, use_container_width=True)

		elif type_of_plot == "Area Plot":
			cols = st.multiselect("Select Columns", all_columns)
			if st.button("Generate Area Plot"):
				df.plot.area(y=cols,alpha=0.4,figsize=(12, 6))
				st.pyplot()



if __name__ == '__main__':
	main()