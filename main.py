import re
import os 
import pandas as pd 
import seaborn as sns
import miceforest as mf
import matplotlib.pyplot as plt 

from pandas import DataFrame
from typing import List, Tuple
from os.path import isdir, join
from scipy.stats import chi2_contingency

title_params = {'fontweight':'bold', 'fontsize':14}
month_encoder = {'Dec':12, 'Nov':11, 'Oct':10, 'Sep':9, 'Aug':8, 'Jul':7, 'Jun':6, 'May':5, 'Apr':4, 
                 'Mar':3, 'Feb':2, 'Jan':1}
grade_encoder = {'B':2, 'C':3, 'A':1, 'E':5, 'F':6, 'D':4, 'G':7}
emp_encoder = {'10+ years':11, '< 1 year':1, '1 year':2, '3 years':4, '8 years':9, '9 years':10, 
               '4 years':5,'5 years':6, '6 years':7, '2 years':3, '7 years':8}
year_encoder = {'2011':5, '2010':4, '2009':3, '2008':2, '2007':1, '2013':7, '2012':6, '2014':8, '2015':9}
axis_params = {'fontsize':12}
random_state = 12345

def missing_data(df:DataFrame, plot_path:str)->None: 
    """Function to compute the missing data in dataset

    Args:
        df (DataFrame): dataset
        plot_path (str): path for plot folder
    """    
    # compute percentage of data missing
    missing_data = (1 - (df.count()/len(df))) *100
    fig = plt.figure(figsize=(10,10))
    plt.barh(missing_data.index.tolist(), missing_data.values.tolist())
    fig.suptitle('Percentage of Missing Data', **title_params)
    plt.xlabel('Percentages [%]', **axis_params)
    plt.ylabel('Field Names', **axis_params)
    plt.grid()
    plt.tight_layout()
    fig.savefig(join(plot_path, 'missing_data.jpeg'))
    plt.close()
    return

def plot_desc_stats(df:DataFrame, num_cols:List, plot_path:str, name:str)->None:
    """Function to plot numerical summary statistics

    Args:
        df (DataFrame): dataset
        num_cols (List): numerical columns
        plot_path (str): path for plot folder
        name (str): plot name
    """    
    # instantiate the descriptive statistitcs
    df_desc = df.describe()
    df_desc.drop(['count'], axis=0, inplace=True)

    # generate descriptive statistics plot for numerical fields
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,10))
    for idx, col in enumerate(num_cols): 
        ax[idx].plot(df_desc[col], marker='.', markersize=20)
        ax[idx].set_title(col, **title_params)
        ax[idx].grid()
    fig.suptitle('Numerical Field Summary Statistics', **title_params)
    fig.savefig(join(plot_path, f'{name}.jpeg'))
    plt.close()
    return

def remove_noise(df:DataFrame)-> DataFrame:
    """Function to filter incorrect data based on assumptions

    Args:
        df (DataFrame): dataset

    Returns:
        DataFrame: filtered dataset
    """    
    df = df.loc[(df['loan_amnt'] > 0) & (df['int_rate'] > 0) & (df['dti'] > 0) & (df['annual_inc'] >=0)]
    return df

def get_boxplots(df:DataFrame, num_cols:List, plot_path:str, name:str)->None:
    """Function to generate the box plots for numerical fields

    Args:
        df (DataFrame): dataset
        num_cols (List): numerical columns
        plot_path (str): path for plot folder
        name (str): plot name
    """    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15,10))
    for idx, col in enumerate(num_cols):
        df.plot.box(column = col, ax=axes[idx])
    fig.suptitle('Numerical Field Box Plots', **title_params)
    plt.tight_layout()
    fig.savefig(join(plot_path, f'{name}.jpeg'))
    plt.close()
    return

def remove_outliers(df:DataFrame, num_cols:List)->DataFrame:
    """Function to filter out the outliers identified by IQR

    Args:
        df (DataFrame): dataset
        num_cols (List): numerical columns

    Returns:
        DataFrame: filtered dataset
    """    
    for col in num_cols:
        q1, q3 = df[col].quantile(.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col][(q1- 1.5*iqr < df[col]) & (df[col] < q3  + 1.5*iqr)]
    return df

def split_date(text:str) -> Tuple[str, str]:
    """Function to split the issue date column into months and years

    Args:
        text (str): issue date

    Returns:
        Tuple[str, str]: month and year of issue date
    """    
    split = text.split('-')
    return split[0], split[1]

def get_bar_plots(df:DataFrame, cat_cols:List, plot_path:str, name:str)->None: 
    """Function to generate the bar plots

    Args:
        df (DataFrame): dataset
        cat_cols (List): categorical columns
        plot_path (str): path for plot folder
        name (str): plot name
    """    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(17,12))
    for idx, col in enumerate(cat_cols):
        df[col].value_counts().plot.barh(ax=axes[idx//3, idx%3])
        axes[idx//3, idx%3].set_xlabel('Count', **axis_params)
        axes[idx//3, idx%3].set_ylabel(col, **axis_params)
        axes[idx//3, idx%3].grid()
    fig.suptitle('Categorical Field Distribution', **title_params)
    plt.tight_layout()
    fig.savefig(join(plot_path, f'{name}.jpeg'))
    plt.close()
    return

def get_heatmap(df:DataFrame, plot_path:str, name:str)->None:
    """Function to generate heatmap of correlation between numerical fields

    Args:
        df (DataFrame): dataset
        plot_path (str): path for plot folder
        name (str): plot name
    """    
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.title('Numerical Fields Correlation Matrix', **title_params)
    plt.savefig(join(plot_path, f'{name}.jpeg'))
    plt.close()
    return

def get_chi_square(df:DataFrame, cat_cols:List, plot_path:str, name:str)->None:
    """Function to compute the chi-square test between each pair of categorical 
    variables 

    Args:
        df (DataFrame): dataset
        cat_cols (List): categorical column list
        plot_path (str): path for plot folder
        name (str): plot name
    """    
    chi_dict = {}
    for i in range(len(cat_cols)-1):
        chi_dict[cat_cols[i]] = {}
        for j in range(i+1, len(cat_cols)):
            cont_table = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
            _, p, _, _ = chi2_contingency(cont_table)
            chi_dict[f'{cat_cols[i]}'][f'{cat_cols[j]}'] = float(p)
    df_chi = pd.DataFrame.from_dict(chi_dict)
    fig = plt.figure(figsize=(10,10))
    sns.heatmap(df_chi, annot=True)
    plt.tight_layout()
    fig.suptitle('Categorical Fields Correlation Matrix', **title_params)
    plt.savefig(join(plot_path, f'{name}.jpeg'))
    plt.close()
    return

def get_imputation(df:DataFrame)->DataFrame:
    """Function to perform imputation on dataset

    Args:
        df (DataFrame): dataset

    Returns:
        DataFrame: imputed dataset
    """    
    #  instantiate the miceforest imputer 
    kds = mf.ImputationKernel(data=df, random_state=random_state)
    # run the miceforest imputer
    kds.mice(iterations=5)
    df_imp = kds.complete_data()
    return df_imp


def main(data_path:str, plot_path:str)-> None: 
    """Function to run the main logic

    Args:
        data_path (str): path for data folder
        plot_path (str): path for plot folder
    """    
    # load the csv file into Dataframe 
    df = pd.read_csv(data_path, index_col=0)

    # seperate the numerical from categorical columns 
    cols = df.columns.tolist()
    num_cols = [f for f in cols if df[f].dtypes != 'object']

    # generate initial plots 
    missing_data(df, plot_path) # generate plot for missing data
    plot_desc_stats(df, num_cols, plot_path, name='init_desc_stats') # generate initial summary stats
    df = remove_noise(df) # remove the noisy data
    plot_desc_stats(df, num_cols, plot_path, name='final_desc_stats') # generate update summary stats
    get_boxplots(df, num_cols, plot_path, name='init_box_plots') # generate the box plots 
    df = remove_outliers(df, num_cols) # reomve outlier data

    # split the issue date into months and years
    df['issue_month'], df['issue_year'] = zip(*df['issue_d'].map(split_date))
    df.drop(columns=['issue_d', 'emp_title'], inplace=True)
    cat_cols = [f for f in df.columns.tolist() if f not in num_cols] # instantiate the categorical columns 

    # generate the plots
    get_bar_plots(df, cat_cols, plot_path, name='init_bar')
    get_heatmap(df[num_cols], plot_path, name='heat_map')
    get_chi_square(df, cat_cols, plot_path, name='chi_square_heatmap')

    # reset the index and impute missing values
    df.reset_index(drop=True, inplace=True)

    # encode the dataset to numerical values 
    ordinal_list = ['issue_month', 'issue_year', 'emp_length', 'grade']
    nominal_list = [f for f in cat_cols if f not in ordinal_list]
    df = pd.get_dummies(df, columns=nominal_list, drop_first=True, dtype=int) # one hot encode nominal values
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace({'grade':grade_encoder, 'issue_month':month_encoder,
                         'issue_year':year_encoder, 'emp_length':emp_encoder}).infer_objects(copy=False) # encode ordinal values
    
    # modify the field names for imputation
    df.columns = df.columns.str.replace('[\s]', '', regex=True).str.replace('[.:()-]', '_', regex=True)
    
    # # perform data imputation
    df = get_imputation(df)
    df.to_csv(join(data_path,'processed.csv'))

    return

if __name__ == '__main__':
    cwd = os.getcwd() # get the current working directory
    data_path = join(cwd, join('data', 'loans_fs.csv')) # get the file path
    plot_path = join(cwd, 'plots') # create the plot path
    if not isdir(plot_path): os.mkdir(plot_path) # create the folder
    main(data_path, plot_path)
