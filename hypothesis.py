import os 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from pandas import DataFrame

title_params = {'fontweight':'bold', 'fontsize':14}
axis_params = {'fontsize':12}

def scatter_plot(df:DataFrame, x:str, y:str, plot_path:str, name:str)->None:
    fig = plt.figure(figsize=(7,7))
    sns.jointplot(x=x, y=y, data=df)
    plt.xlabel(x, **axis_params)
    plt.ylabel(y, **axis_params)
    plt.title(f'Relationship between {x} & {y}')
    plt.tight_layout()
    plt.grid()
    plt.savefig(join(plot_path, f'{name}.jpeg'))
    plt.close()
    return

def main(data_path:str, plot_path:str)->None: 
    # load processed dataset
    df = pd.read_csv(data_path, index_col=0)
    
    # generate the scatter plots 
    scatter_plot(df, x='dti', y='grade', plot_path=plot_path, name='dti_grade')
    scatter_plot(df, x='emp_length', y='grade', plot_path=plot_path, name='emplen_grade')
    return

if __name__ == '__main__':
    cwd = os.getcwd()
    data_path = join(cwd, 'data/processed.csv')
    plot_path = join(cwd, 'plots')
    main(data_path, plot_path)