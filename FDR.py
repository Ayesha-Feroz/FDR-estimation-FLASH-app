import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

def generate_plots(combined_df, original_df, ppmtol):
    try:
        # Create the fdval dataframe by combining the ScanNum and MonoisotopicMass columns from the combined_df
        fdval = combined_df[['ScanNum', 'MonoisotopicMass']]

        # Convert the ScanNum column to integers
        fdval['ScanNum'] = fdval['ScanNum'].astype(int)

        # Create the tpindex1 logical index by checking if the values in the ScanNum column of fdval are present in the Scan column of original_df
        tpindex1 = fdval['ScanNum'].isin(original_df['ScanNum'].dropna().replace([np.inf, -np.inf], np.nan).astype(int))

        # Adjusted code to allow for ±1 Da error with user-defined ppmtol Da tolerance
        tpindex2 = np.logical_or.reduce([
            (fdval.iloc[:, 1].values[:, np.newaxis] >= original_df.iloc[:, 2].values - ppmtol / 10) &
            (fdval.iloc[:, 1].values[:, np.newaxis] <= original_df.iloc[:, 2].values + ppmtol / 10),
            (fdval.iloc[:, 1].values[:, np.newaxis] >= original_df.iloc[:, 2].values - (ppmtol + 1) / 10) &
            (fdval.iloc[:, 1].values[:, np.newaxis] <= original_df.iloc[:, 2].values - (ppmtol - 1) / 10),
            (fdval.iloc[:, 1].values[:, np.newaxis] >= original_df.iloc[:, 2].values + (ppmtol - 1) / 10) &
            (fdval.iloc[:, 1].values[:, np.newaxis] <= original_df.iloc[:, 2].values + (ppmtol + 1) / 10)
        ])

        # Create the tpindex logical index by combining tpindex1, tpindex2, and the DummyIndex column from combined_df with values of 0
        tpindex = np.logical_and.reduce((
            tpindex1.values.flatten(),
            tpindex2.any(axis=1),
            combined_df['TargetDecoyType'].values == 0
        ))

        # Identify false positives and decoys
        fpindex = np.logical_and.reduce((
            np.logical_not(tpindex),
            combined_df['TargetDecoyType'].values == 0
        ))
        decoyindex = combined_df['TargetDecoyType'] > 0

        # Identify true positives, false positives, and decoys
        true_positives = combined_df.loc[tpindex]
        false_positives = combined_df.loc[fpindex]
        decoy_masses = combined_df.loc[decoyindex]

        # Count the number of instances for each category
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        decoy_count = len(decoy_masses)

        # Plotting the histograms, ECDF, and KDE
        st.subheader('Histogram of Qscore')
        fig_hist = plt.figure(figsize=(8, 6))
        plt.hist(combined_df.loc[fpindex, 'Qscore'], bins=100, alpha=0.7, label='False Positive Masses', color='red',
                 edgecolor='grey')
        plt.hist(combined_df.loc[tpindex, 'Qscore'], bins=100, alpha=0.6, label='True Positive Masses', color='green',
                 edgecolor='grey')
        plt.hist(combined_df.loc[decoyindex, 'Qscore'], bins=100, alpha=0.4, label='Decoy Masses', color='blue',
                 edgecolor='grey')
        plt.xlabel('Qscore')
        plt.ylabel('Count')
        plt.legend()
        st.pyplot(fig_hist)

        st.subheader('Empirical Cumulative Distribution Function')
        fig_ecdf = plt.figure(figsize=(10, 6))
        sns.ecdfplot(data=combined_df.loc[tpindex, 'Qscore'], label='True Positive Masses', color='green')
        sns.ecdfplot(data=combined_df.loc[fpindex, 'Qscore'], label='False Positive Masses', color='red')
        sns.ecdfplot(data=combined_df.loc[decoyindex, 'Qscore'], label='Decoy Masses', color='blue')
        plt.xlabel('Qscore')
        plt.ylabel('ECDF')
        plt.legend()
        st.pyplot(fig_ecdf)

        st.subheader('Kernel Density Estimate')
        fig_kde = plt.figure(figsize=(10, 6))
        sns.kdeplot(data=combined_df.loc[tpindex, 'Qscore'], label='True Positive Masses', shade=True, color='green')
        sns.kdeplot(data=combined_df.loc[fpindex, 'Qscore'], label='False Positive Masses', shade=True, color='red')
        sns.kdeplot(data=combined_df.loc[decoyindex, 'Qscore'], label='Decoy Masses', shade=True, color='blue')
        plt.xlabel('Qscore')
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(fig_kde)
        # Pie chart for true positives, false positives, and decoy masses
        st.subheader('Pie Chart: True Positives, False Positives, Decoy Masses')
        fig_pie, ax_pie = plt.subplots()
        labels = ['True Positives', 'False Positives', 'Decoy Masses']
        sizes = [tp_count, fp_count, decoy_count]
        colors = ['green', 'red', 'blue']
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig_pie)

        st.subheader('Distribution Plot: Different DecoyTypes')
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=combined_df.loc[combined_df['TargetDecoyType'] == 1, 'Qscore'], label='Noise Decoys',
                    shade=True)
        sns.kdeplot(data=combined_df.loc[combined_df['TargetDecoyType'] == 2, 'Qscore'], label='Isotope Decoys',
                    shade=True)
        sns.kdeplot(data=combined_df.loc[combined_df['TargetDecoyType'] == 3, 'Qscore'], label='Charge Decoys',
                    shade=True)
        plt.xlabel('Qscore')
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(fig_dist)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Sidebar for file uploads


st.sidebar.header("File Uploads")
combined_file = st.sidebar.file_uploader("Upload Combined File (CSV)", type=["csv"])
original_file = st.sidebar.file_uploader("Upload Original File (CSV)", type=["csv"])

# Sidebar for other options
ppmtol = st.sidebar.slider("Set ppmtol", min_value=1, max_value=20, value=5)

# Title
st.title('FDR Estimation FLASH App 🚀')

# Subheader
st.subheader(f'Checkout distributions plots for different tolerance: {ppmtol} ppm Tolerance')

# Check if files are uploaded
if combined_file is not None and original_file is not None:
    try:
        combined_df = pd.read_csv(combined_file)
        original_df = pd.read_csv(original_file)

        # Check if dataframes are not empty
        if combined_df.empty or original_df.empty:
            st.error("Uploaded files are empty. Please upload files with data.")
        else:
            # Subheader and button to generate plots
            st.subheader('Generate Plots')
            if st.button('Generate Plots'):
                generate_plots(combined_df, original_df, ppmtol)

    except pd.errors.EmptyDataError:
        st.error("Error reading the uploaded files. Please make sure the files are in CSV format and contain data.")
