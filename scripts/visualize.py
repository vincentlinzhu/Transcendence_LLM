import pickle
from evaluate import load
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec



def load_and_calculate(squad_metric, temperatures, model_names, num_bootstrap_samples):
    for model_name in model_names:
        data_per_temp = {}
        for t in temperatures:
            # Load Data
            file_path = f"scripts/evaluation_history_model='{model_name}'_temp={t}_total_evals=1000.pkl"
            data_per_temp[t] = pickle.load(
                open(file_path, "rb")
            )
        
        bootstrap_results = defaultdict(list) # dict of lists, lists are 1000 metrics (includes F1 and EM)
        for i in tqdm(range(num_bootstrap_samples)):
            length_of_data = len(data_per_temp[0.001])
            random_sample_indices = np.random.choice(length_of_data, size=10, replace=True)
            for temp, data in data_per_temp.items():
                #bootstrap with 1000 resamples
                dataset = [data[i] for i in random_sample_indices]
                correct_count = 0
                # process generation and run metric on it
                predictions = []
                references = []
                for example in dataset:
                    if example['correct'] == 'yes':
                        correct_count += 1
                    processed_answer = example['generation'].strip().lower()
                    reference_answers = example['input_correct_responses']
                    predictions.append({"id": example['id'], "prediction_text": processed_answer, 'no_answer_probability': 0.0})
                    references.append({"id": example['id'], "answers": {"answer_start": [0], "text": reference_answers}})
                metric_result = squad_metric.compute(predictions=predictions, references=references)
                correct_percentage = correct_count / len(dataset)
                metric_tuple = (metric_result, correct_percentage)
                bootstrap_results[temp].append(metric_tuple)
                
        # with open(f"metrics_{model_name}_{num_bootstrap_samples}.pkl", "wb") as fin:
        with open(f"metrics_{model_name}_10_1000.pkl", "wb") as fin:
            pickle.dump(bootstrap_results, fin)
    
    
def visualize_three_plots(model_name, num_bootstrap_samples):
    # for model_name in model_names:
    bootstrap_results = pickle.load(
        # open(f"metrics_{model_name}_{num_bootstrap_samples}.pkl", "rb")
        open(f"metrics_{model_name}_10_1000.pkl", "rb")
    )
    # Process Results
    desired_data_f1 = defaultdict(list)
    desired_data_em = defaultdict(list)
    desired_data_pm = defaultdict(list)
    for temp, metrics_list in bootstrap_results.items():
        for metric_tuple in metrics_list:
            desired_data_f1[temp].append(metric_tuple[0]['f1'])
            desired_data_em[temp].append(metric_tuple[0]['exact'])
            desired_data_pm[temp].append(metric_tuple[1] * 100)

    # Create a DataFrame
    df_f1 = pd.DataFrame(desired_data_f1)
    df_em = pd.DataFrame(desired_data_em)
    df_pm = pd.DataFrame(desired_data_pm)
    print("F1 Dataframe\n", df_f1)
    print("Exact Match Dataframe\n", df_em)
    print("Semantic Match Dataframe\n", df_pm)
    
    with sns.axes_style("darkgrid"):
        plt.rcParams.update(
            {
                "font.size": 20,
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsfonts}",
                "font.family": "Computer Modern Serif",
            }
        )
        palette = sns.color_palette("viridis", 3)
        fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=False)
        plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.15, wspace=0.2)
        
        if model_name == 'gemma2:2b-text-fp16':
            title = f'Gemma 2 (2B Params) SQuADv2 Evaluation Metrics'
        if model_name == 'gpt2':
            title = f'GPT-2 (163M Params) SQuADv2 Evaluation Metrics'
        if model_name == 'mistral:text':
            title = f'Mistral (7B Params) SQuADv2 Evaluation Metrics'
        if model_name == 'qwen2:7b-text':
            title = f'Qwen2 (7B Params) SQuADv2 Evaluation Metrics'
        fig.suptitle(title, fontsize=30, y=0.97)
        
        ##########################################################################
        # F1 Score vs Temperature
        ##########################################################################
        melted_df = df_f1.melt(var_name='Temperature', value_name='F1 Score')
        std_f1 = melted_df['F1 Score'].std()
        ci = 1.96 * std_f1 / np.sqrt(1000)
        sns.lineplot(data=melted_df, x='Temperature', y='F1 Score', color=palette[2], ax=axes[0])
        axes[0].set_title('F1-Score vs Temperature')
        x_ticks=melted_df['Temperature'].unique() 
        axes[0].set_xticks(x_ticks)
        axes[0].set_xticklabels([f"{temp:.15g}" for temp in x_ticks])  # Format temperature labels
        axes[0].legend().set_visible(False)
        axes[0].set_xlabel(r"Temperature ($\tau$)")
        axes[0].set_ylabel("F1-Score")
        axes[0].grid(True)
        
        ##########################################################################
        # Exact Matches vs Temperature
        ##########################################################################
        melted_df = df_em.melt(var_name='Temperature', value_name='Exact Match Percentage')
        sns.lineplot(data=melted_df, x='Temperature', y='Exact Match Percentage', color=palette[1], ax=axes[1])    # Set x-scale to log to match the user's example
        axes[1].set_title('Exact Match % vs Temperature')
        x_ticks=melted_df['Temperature'].unique() 
        axes[1].set_xticks(x_ticks)
        axes[1].set_xticklabels([f"{temp:.15g}" for temp in x_ticks])  # Format temperature labels
        axes[1].legend().set_visible(False)
        axes[1].set_xlabel(r"Temperature ($\tau$)")
        axes[1].set_ylabel(r"Exact Match (\%)")
        axes[1].grid(True)
        
        ##########################################################################
        # Semantic Matches vs Temperature
        ##########################################################################
        melted_df = df_pm.melt(var_name='Temperature', value_name='Semantic Match Percentage')
        sns.lineplot(data=melted_df, x='Temperature', y='Semantic Match Percentage', color=palette[0], ax=axes[2])    # Set x-scale to log to match the user's example
        axes[2].set_title('Semantic Match % vs Temperature')
        x_ticks=melted_df['Temperature'].unique() 
        axes[2].set_xticklabels([f"{temp:.15g}" for temp in x_ticks])  # Format temperature labels
        axes[2].set_xticks(x_ticks)
        axes[2].legend().set_visible(False)
        axes[2].set_xlabel(r"Temperature ($\tau$)")
        axes[2].set_ylabel(r"Semantic Match (\%)")
        axes[2].grid(True)
        
        plt.savefig(f"{title}.png")
        

def get_model_display_name(model_name):
    if model_name == 'mistral:text':
        return 'Mistral (7B)'
    elif model_name == 'gemma2:2b-text-fp16':
        return 'Gemma 2 (2B)'
    elif model_name == 'gpt2':
        return 'GPT-2 (163M)'
    elif model_name == 'qwen2:7b-text':
        return 'Qwen2 (7B)'
    else:
        return model_name.split(':')[0].capitalize()


def visualize_combined_plots_no_legend(model_names, num_bootstrap_samples):
    plt.rcParams.update({
        "font.size": 25,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Computer Modern Serif",
    })
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(1, 3, figsize=(36, 8), sharey=False)
        plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.15, wspace=0.2)
        fig.suptitle('SQuADv2 Evaluation Comparison', fontsize=40, y=0.96)
        
        palette = sns.color_palette("viridis", len(model_names))
        
        metrics = ['F1 Score', r'Exact Match (\%)', r'Semantic Match (\%)']
        
        lines = []
        labels = []
        
        for idx, metric in enumerate(metrics):
            for model_idx, model_name in enumerate(model_names):
                bootstrap_results = pickle.load(open(f"metrics_{model_name}_10_1000.pkl", "rb"))
                
                desired_data = defaultdict(list)
                for temp, metrics_list in bootstrap_results.items():
                    for metric_tuple in metrics_list:
                        if metric == 'F1 Score':
                            desired_data[temp].append(metric_tuple[0]['f1'])
                        elif metric == r'Exact Match (\%)':
                            desired_data[temp].append(metric_tuple[0]['exact'])
                        elif metric == r'Semantic Match (\%)':
                            desired_data[temp].append(metric_tuple[1] * 100)
                
                df = pd.DataFrame(desired_data)
                melted_df = df.melt(var_name='Temperature', value_name=metric)
                
                display_name = get_model_display_name(model_name)
                line = sns.lineplot(data=melted_df, x='Temperature', y=metric, 
                                    color=palette[model_idx], ax=axes[idx])
                
                if idx == 0:  # Only add to legend once
                    lines.append(line.lines[-1])
                    labels.append(display_name)
            
            axes[idx].set_title(f'{metric} vs Temperature')
            axes[idx].set_xlabel(r"Temperature ($\tau$)")
            axes[idx].set_ylabel(metric)
            axes[idx].set_yticks(range(0, 80, 10))
            axes[idx].set_yticklabels(["0","10","...","30","40","50","60","70"]) 
            axes[idx].grid(True)
            # if idx == 2:
            axes[idx].legend(title="Models")
            x_ticks = melted_df['Temperature'].unique() 
            axes[idx].set_xticks(x_ticks)
            axes[idx].set_xticklabels([f"{temp:.3g}" for temp in x_ticks])
        
        # Create a single legend outside the plots
        # fig.legend(lines, labels, loc='center right', title="Models", bbox_to_anchor=(0.95, 0.5))
    
    plt.savefig("Combined_SQuADv2_Evaluation_Metrics.png", dpi=300, bbox_inches='tight')


def visualize_combined_plots(model_names, num_bootstrap_samples):
    plt.rcParams.update({
        "font.size": 35,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Computer Modern Serif",
    })
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(1, 3, figsize=(32, 10), sharey=False)
        plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.15, wspace=0.18)
        # fig.suptitle('SQuADv2 Evaluation Metrics Comparison', fontsize=40, y=0.96)
        
        palette = sns.color_palette("viridis", len(model_names))
        palette = palette[::-1]
        
        metrics = ['F1 Score', r'Exact Match (\%)', r'Semantic Match (\%)']
        
        for idx, metric in enumerate(metrics):
            for model_idx, model_name in enumerate(model_names):
                bootstrap_results = pickle.load(open(f"metrics_{model_name}_10_1000.pkl", "rb"))
                
                desired_data = defaultdict(list)
                for temp, metrics_list in bootstrap_results.items():
                    for metric_tuple in metrics_list:
                        if metric == 'F1 Score':
                            desired_data[temp].append(metric_tuple[0]['f1'])
                        elif metric == r'Exact Match (\%)':
                            desired_data[temp].append(metric_tuple[0]['exact'])
                        elif metric == r'Semantic Match (\%)':  # Semantic Match Percentage
                            desired_data[temp].append(metric_tuple[1] * 100)
                
                df = pd.DataFrame(desired_data)
                melted_df = df.melt(var_name='Temperature', value_name=metric)
                
                display_name = get_model_display_name(model_name)
                sns.lineplot(data=melted_df, x='Temperature', y=metric, 
                            color=palette[model_idx], ax=axes[idx], 
                            label=display_name)
            
            axes[idx].set_title(f'{metric} vs Temperature', pad=20)
            axes[idx].set_xlabel(r"Temperature ($\tau$)")
            axes[idx].set_ylabel(metric)
            axes[idx].set_yticks(range(0, 80, 10))
            axes[idx].set_yticklabels(["0","10","...","30","40","50","60","70"])
            axes[idx].grid(True)
            if idx == 0:
                axes[idx].legend(title="Models", fontsize=24, bbox_to_anchor=(0.45, 0.6))
            elif idx == 1:
                axes[idx].legend(title="Models", fontsize=24, bbox_to_anchor=(0.45, 0.35))
            elif idx == 2:
                axes[idx].legend(title="Models", fontsize=24, bbox_to_anchor=(0.45, 0.4))
            # else:
            #     axes[idx].legend().set_visible(False)
                
            
            x_ticks = melted_df['Temperature'].unique() 
            axes[idx].set_xticks(x_ticks)
            axes[idx].set_xticklabels([f"{temp:.3g}" for temp in x_ticks])
    
    plt.savefig("Combined_SQuADv2_Evaluation_Metrics.png", dpi=300, bbox_inches='tight')


def visualize_combined_plots_broken_axis(model_names, num_bootstrap_samples):
    plt.rcParams.update({
        "font.size": 35,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Computer Modern Serif",
    })
    
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(32, 7))
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1], hspace=0.1)
        
        palette = sns.color_palette("viridis", len(model_names))
        palette = palette[::-1]
        
        metrics = ['F1 Score', r'Exact Match (\%)', r'Semantic Match (\%)']
        
        for idx, metric in enumerate(metrics):
            ax_top = fig.add_subplot(gs[0, idx])
            ax_bottom = fig.add_subplot(gs[1, idx])
            axes = [ax_top, ax_bottom]
            ax_bottom.set_zorder(1)  # Ensure the bottom axes are behind the legend
            
            for model_idx, model_name in enumerate(model_names):
                bootstrap_results = pickle.load(open(f"metrics_{model_name}_10_1000.pkl", "rb"))
                
                desired_data = defaultdict(list)
                for temp, metrics_list in bootstrap_results.items():
                    for metric_tuple in metrics_list:
                        if metric == 'F1 Score':
                            desired_data[temp].append(metric_tuple[0]['f1'])
                        elif metric == r'Exact Match (\%)':
                            desired_data[temp].append(metric_tuple[0]['exact'])
                        elif metric == r'Semantic Match (\%)':
                            desired_data[temp].append(metric_tuple[1] * 100)
                
                df = pd.DataFrame(desired_data)
                melted_df = df.melt(var_name='Temperature', value_name=metric)
                
                display_name = get_model_display_name(model_name)
                for ax in axes:
                    sns.lineplot(data=melted_df, x='Temperature', y=metric, 
                                color=palette[model_idx], ax=ax, 
                                label=display_name)
            
            # Set y-axis limits and ticks
            if idx == 1: 
                ax_top.set_ylim(22, 65)
                ax_bottom.set_ylim(-.2, 1.2)
                ax_top.set_yticks(np.arange(30, 70, 10))
            else:
                ax_top.set_ylim(25, 75)
                ax_bottom.set_ylim(-1, 7)
                ax_top.set_yticks(np.arange(30, 80, 10))
                
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.xaxis.tick_top()
            ax_top.tick_params(labeltop=False)
            ax_bottom.xaxis.tick_bottom()
            ax_top.set_xlabel(None)
            
            # Add broken axis lines
            d = .015  # size of diagonal lines
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            
            # Set labels and title
            ax_bottom.set_xlabel(r"Temperature ($\tau$)")
            ax_bottom.set_ylabel(None)
            ax_top.yaxis.set_label_coords(-0.08, 0.35)  # Adjust these values as needed
            # ax_bottom.set_ylabel(metric)
            ax_top.set_title(f'{metric} vs Temperature', pad=25)
            
            # Set x-ticks
            x_ticks = melted_df['Temperature'].unique() 
            ax_bottom.set_xticks(x_ticks)
            ax_bottom.set_xticklabels([f"{temp:.3g}" for temp in x_ticks])
            
            # Add legend to the first subplot only
            leg = ax_top.legend(fontsize=20, bbox_to_anchor=(0.2, 0.45), loc='upper center')
            if idx == 1:
                leg = ax_top.legend(fontsize=20, bbox_to_anchor=(0.2, 0.42), loc='upper center')
            ax_bottom.legend().set_visible(False)
            leg.set_zorder(100)
                # ax_bottom.legend(title="Models", fontsize=24, bbox_to_anchor=(0.45, -0.15))
            # else:
            #     ax_top.legend().set_visible(False)
            #     ax_bottom.legend().set_visible(False)
    
    # plt.tight_layout()
    plt.savefig("Combined_SQuADv2_Evaluation_Metrics_Broken_Axis.png", dpi=300, bbox_inches='tight')


def main():
    num_bootstrap_samples = 1000
def main():
    squad_metric = load("squad_v2")
    temperatures = np.array([0.001, 0.25, 0.5, 0.75, 1, 1.5])
    model_names = ['qwen2:7b-text', 'mistral:text', 'gemma2:2b-text-fp16', 'gpt2']
    num_bootstrap_samples = 1000
    
    load_and_calculate(squad_metric, temperatures, model_names, num_bootstrap_samples)
    # visualize_combined_plots(model_names, num_bootstrap_samples)
    visualize_combined_plots_broken_axis(model_names, num_bootstrap_samples)

    
if __name__ == "__main__":
    main()