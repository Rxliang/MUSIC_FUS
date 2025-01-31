import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from mpl_toolkits.mplot3d import Axes3D, art3d  # Import 3D plotting toolkit
from scipy import stats
import re
from sklearn.preprocessing import MinMaxScaler

class MUSIC_TUS_DataAnalyzer:
    """
    A class to analyze and visualize ultrasound data from .mat files.

    Attributes:
        root_dir (str): The root directory containing the data.
        date_select (str or list): Specific date(s) to select for analysis.
        exclusion_dict (dict): Dictionary specifying dates, sessions, or files to exclude.
        verbose (bool): If True, enables detailed logging.
        data_keys (list): List of keys to look for in the .mat files.
    """

    def __init__(self, root_dir, exclusion_dict=None, verbose=False, date_select=None, data_keys=None):
        self.root_dir = root_dir
        self.date_select = date_select
        self.exclusion_dict = exclusion_dict or {}
        self.verbose = verbose
        self.data_keys = data_keys or ["US_chunk", "US_chunk_1", "US1"]
        self.subdates = []
        self.sessions = []
        self.files = []
        self.scaler = {}
        self.raw_data = {}
        self.processed_data = pd.DataFrame()
        self.logger = self.setup_logger()
        self.load_data()
        self.process_data()

    def setup_logger(self):
        """Sets up the logger for the class."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self):
        """Loads data from the directory structure into raw_data dictionary."""
        # Get list of date directories
        self.subdates = [
            subdir for subdir in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, subdir))
        ]

        # Filter dates if date_select is specified
        if self.date_select:
            if isinstance(self.date_select, str):
                self.subdates = [subdir for subdir in self.subdates if self.date_select in subdir]
            elif isinstance(self.date_select, list):
                self.subdates = [subdir for subdir in self.subdates if any(ds in subdir for ds in self.date_select)]
            else:
                raise TypeError("date_select must be a string or a list of strings")

        # Exclude dates based on exclusion_dict
        if 'dates' in self.exclusion_dict:
            self.subdates = [d for d in self.subdates if d not in self.exclusion_dict['dates']]

        # Get sessions within each date
        for date in self.subdates:
            date_path = os.path.join(self.root_dir, date)
            self.scaler[date] = []
            sessions = [
                subdir for subdir in os.listdir(date_path)
                if os.path.isdir(os.path.join(date_path, subdir))
            ]
            # Exclude sessions if specified
            if 'sessions' in self.exclusion_dict and date in self.exclusion_dict['sessions']:
                sessions = [s for s in sessions if s not in self.exclusion_dict['sessions'][date]]
            self.sessions.extend([(date, session) for session in sessions])

        # Get .mat files within each session
        for date, session in self.sessions:
            session_path = os.path.join(self.root_dir, date, session)
            files = [
                file for file in os.listdir(session_path) if file.endswith('.mat')
            ]
            # Exclude files if specified
            if 'files' in self.exclusion_dict and date in self.exclusion_dict['files'] \
                    and session in self.exclusion_dict['files'][date]:
                files = [
                    f for f in files if f not in self.exclusion_dict['files'][date][session]
                ]
            self.files.extend([(date, session, file) for file in files])

    def load_file_data(self, key):
        """Lazy loads data from a specific file."""
        if key in self.raw_data:
            return self.raw_data[key]

        date, session, file = key
        file_path = os.path.join(self.root_dir, date, session, file)
        try:
            data = scipy.io.loadmat(file_path)
            self.raw_data[key] = data
            if self.verbose:
                self.logger.info(f"Loaded data from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None


    def process_data(self, remove_outliers=True, replace_zeros=True, smoothing=True, clip_range=(None, None), jump_threshold=500):
        """
        Processes raw data and stores results in processed_data DataFrame.
        Also handles sudden jumps and stores the filtered time series data in the 'filtered_data' column.

        Parameters:
            remove_outliers: bool, whether to remove extreme outliers based on Z-score.
            replace_zeros: bool, whether to replace zeros in the data.
            smoothing: bool, whether to apply a smoothing filter to the data.
            clip_range: tuple, the lower and upper bounds to clip extreme values (None means no clipping).
            jump_threshold: float, the threshold for detecting and removing sudden jumps in the data.
        """
        records = []
        for date, session, file in self.files:
            key = (date, session, file)
            data = self.load_file_data(key)
            if data is None:
                continue

            try:
                # Access the time series data with the correct key
                tempt_env = None
                for data_key in self.data_keys:
                    if data.get(data_key) is not None:
                        tempt_env = data[data_key][0][0]
                        print("!!!!!!",tempt_env.shape)
                        break
                if tempt_env is None:
                    self.logger.warning(f"No recognized data key found in {key}")
                    continue  # Skip if none of the keys are found

                # Handle zeros by interpolation
                if replace_zeros:
                    tempt_env[tempt_env == 0] = np.nan  # Mark zeros as NaN
                    tempt_env = pd.Series(tempt_env).interpolate().fillna(method="bfill").fillna(method="ffill").values  # Interpolate and fill NaN

                # Remove extreme outliers using Z-score method
                if remove_outliers:
                    z_scores = np.abs(stats.zscore(tempt_env))
                    tempt_env = np.where(z_scores > 3, np.nan, tempt_env)  # Mark outliers as NaN
                    tempt_env = pd.Series(tempt_env).interpolate().fillna(method="bfill").fillna(method="ffill").values  # Interpolate NaN

                # Handle sudden jumps based on the rate of change
                if jump_threshold is not None:
                    rate_of_change = np.abs(np.diff(tempt_env))
                    jump_mask = np.insert(rate_of_change > jump_threshold, 0, False)  # Mark jumps as True
                    tempt_env = np.where(jump_mask, np.nan, tempt_env)  # Replace large jumps with NaN
                    tempt_env = pd.Series(tempt_env).interpolate().fillna(method="bfill").fillna(method="ffill").values  # Interpolate NaN

                # Apply smoothing filter (e.g., rolling mean)
                if smoothing:
                    tempt_env = pd.Series(tempt_env).rolling(window=50, min_periods=1, center=True).mean().values

                # Clip extreme values (optional)
                if clip_range[0] is not None or clip_range[1] is not None:
                    tempt_env = np.clip(tempt_env, clip_range[0], clip_range[1])

                # Store the filtered data as a new column 'filtered_data'
                filtered_data = tempt_env

                # Calculate means over specified ranges (adjust these ranges as needed)
                mean_100_3000 = np.mean(filtered_data[100:3000])
                mean_8000_10000 = np.mean(filtered_data[8000:10000])
                mean_all = np.mean(filtered_data)

                # Append results to records
                records.append({
                    'date': date,
                    'session': session,
                    'file': file,
                    'mean_100_3000': mean_100_3000,
                    'mean_8000_10000': mean_8000_10000,
                    'mean_all': mean_all,
                    'filtered_data': filtered_data  # Store the filtered data in the DataFrame
                })
                if self.verbose:
                    self.logger.debug(f"Processed data for {date}/{session}/{file}")
            except Exception as e:
                self.logger.error(f"Error processing data for {key}: {e}")

        # Convert records to DataFrame
        self.processed_data = pd.DataFrame(records)
    
    def analyze_data(self):
        """Analyzes processed data to extract trends or perform statistical analysis."""
        if self.processed_data.empty:
            self.logger.warning("Processed data is empty. No analysis will be performed.")
            return

        # Example analysis: Calculate descriptive statistics
        self.analysis_results = self.processed_data.describe()
        if self.verbose:
            self.logger.info("Analysis Results:")
            self.logger.info(f"\n{self.analysis_results}")

    def plot_data(self, save_fig=False, fig_dir='plots'):
        """
        Plots the processed data for each session, stitching the plots of each .mat file in the correct numerical order.
        Gaps are added between the plots of different files to distinguish them.

        Parameters:
            save_fig (bool): If True, saves the plots to files.
            fig_dir (str): Directory to save the plots if save_fig is True.
        """
        if save_fig and not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Define a function to extract numbers from filenames for sorting
        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else float('inf')

        # Group the processed data by session
        grouped_data = self.processed_data.groupby(['date', 'session'])
        
        for (date, session), group in grouped_data:
            # Sort the group by the numerical value in the filenames
            sorted_group = group.sort_values(by='file', key=lambda x: x.map(extract_number))
            # print(sorted_group.head())
            # Initialize variables for stitching plots
            stitched_data = []
            file_names = []
            stitched_indices = []
            current_index = 0

            # Loop through each file in the sorted group
            for _, row in sorted_group.iterrows():
                file_name = row['file']
                # print(file_name)
                file_names.append(file_name)
                filtered_data = row['filtered_data']
                # print()
                
                # Append the data for this file to the stitched data list
                stitched_data.extend(filtered_data)

                # Append the sample indices (with a gap between files)
                stitched_indices.extend(np.arange(current_index, current_index + len(filtered_data)))
                
                current_index += len(filtered_data) + len(filtered_data)//3  # Leave a gap of 500 samples between each file
            # np.save(f"{date}_{session}_stitched.npy", stitched_indices)
            # Plot the stitched data for this session
            plt.figure(figsize=(14, 6))
            
            plt.plot(stitched_indices, stitched_data, label=f'Session: {session}', color='blue')
            print(file_names)
            plt.title(f'Stitched Plot for Session: {session} on {date}')
            plt.xlabel('Sample Index (with gaps)')
            plt.ylabel('Amplitude')
            plt.legend()

            # Save or show the figure
            if save_fig:
                plot_filename = f"{date}_{session}_stitched_plot.png"
                plt.savefig(os.path.join(fig_dir, plot_filename))
                self.logger.info(f"Saved plot to {os.path.join(fig_dir, plot_filename)}")
                plt.close()
            else:
                plt.show()

    def plot_processed_data(self, window_length=1000, save_fig=False, fig_dir='plots', palette='Spectral'):
        """
        Compares the last segment of each data with the first segment of the next data within the same session.
        Collects the full data distributions and plots them using split violin plots for each change within sessions.
        
        Parameters:
            window_length (int): The length of the segments to compare.
            save_fig (bool): If True, saves the plots to files.
            fig_dir (str): Directory to save the plots if save_fig is True.
            palette (str): Color palette for the plots.
        """
        # Initialize lists to store data values, labels, and identifiers
        data_values = []
        segment_labels = []
        change_labels = []
        session_labels = []
        # Group the processed data by session
        grouped_data = self.processed_data.groupby(['date', 'session'])
        for (date, session), group in grouped_data:
    # Sort the group by file name to ensure consistent ordering
            sorted_group = group.sort_values(by='file')
            sorted_keys = list(sorted_group[['date', 'session', 'file']].itertuples(index=False, name=None))
            num_changes = len(sorted_keys) - 1  # Number of changes in this session

            # Iterate over pairs of adjacent data entries within the same session
            for i in range(num_changes):
                key1 = sorted_keys[i]
                key2 = sorted_keys[i + 1]
                change_label = f'TUS {i + 1}'  # Label for the change

                # Access filtered_data for the two consecutive entries
                data1 = self.processed_data[self.processed_data['date'] == key1[0]]
                data1 = data1[data1['session'] == key1[1]]
                data1 = data1[data1['file'] == key1[2]]['filtered_data'].values[0]

                data2 = self.processed_data[self.processed_data['date'] == key2[0]]
                data2 = data2[data2['session'] == key2[1]]
                data2 = data2[data2['file'] == key2[2]]['filtered_data'].values[0]

                if data1 is None or data2 is None:
                    continue


                # Get last segment of data1 and first segment of data2
                last_segment = data1[-window_length:]
                first_segment = data2[:window_length]

                # Collect data values with labels
                data_values.extend(last_segment)
                segment_labels.extend(['Pre'] * window_length)
                change_labels.extend([change_label] * window_length)
                session_labels.extend([session] * window_length)

                data_values.extend(first_segment)
                segment_labels.extend(['Post'] * window_length)
                change_labels.extend([change_label] * window_length)
                session_labels.extend([session] * window_length)

        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            'changes': change_labels,
            'Segment': segment_labels,
            'Value': data_values,
            'Session': session_labels
        })

        if plot_data.empty:
            self.logger.warning("No data available for plotting.")
            return

        if self.verbose:
            self.logger.info("Prepared data for violin plot:")
            self.logger.info(f"\n{plot_data.head()}")

        # Plot violin plots for each session
        unique_sessions = plot_data['Session'].unique()
        if save_fig and not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        for session in unique_sessions:
            session_data = plot_data[plot_data['Session'] == session]
            plt.figure(figsize=(12, 6))
            sns.violinplot(x='changes', y='Value', hue='Segment', data=session_data, split=True, inner="quartile", palette=palette)
            plt.title(f'Violin Plot of Data Distributions by Change within Session {session}')
            plt.xlabel('TUS Session')
            plt.ylabel('Frequency Shift (Hz)')
            plt.legend(title='Segment', loc='upper center', ncol=2, frameon=False)
            if save_fig:
                plot_filename = f"violin_plot_session_{session}.png"
                plt.savefig(os.path.join(fig_dir, plot_filename))
                self.logger.info(f"Saved plot to {os.path.join(fig_dir, plot_filename)}")
                plt.close()
            else:
                plt.show()

    def plot_mean_values_3d(self):
        """
        Plots each .mat file as a point using its mean value as z-value,
        with each session on the y-axis, and x being the mat file index within the session.
        For high Z-values, the fill baseline is set slightly below the minimum Z-value.
        """
        if self.processed_data.empty:
            self.logger.warning("Processed data is empty. Cannot generate 3D plot.")
            return

        # Create a list of unique sessions
        session_list = sorted(self.processed_data['session'].unique())
        
        num_sessions = len(session_list)

        # Determine the maximum number of files in any session
        max_files_per_session = self.processed_data.groupby('session').size().max()

        # Initialize X, Y, Z arrays
        X = np.zeros((num_sessions, max_files_per_session))
        Y = np.zeros((num_sessions, max_files_per_session))
        Z = np.full((num_sessions, max_files_per_session), np.nan)  # Use NaN for missing data

        # Populate X, Y, Z arrays
        for session_idx, session in enumerate(session_list):
            session_data = self.processed_data[self.processed_data['session'] == session]
            session_data = session_data.sort_values('file').reset_index(drop=True)
            num_files = len(session_data)

            X[session_idx, :num_files] = np.arange(num_files)  # File indices
            Y[session_idx, :num_files] = session_idx  # Session index
            Z[session_idx, :num_files] = session_data['mean_all'].values  # Mean values

        # Find the minimum Z value across all data points
        global_min_z = np.nanmin(Z)
        baseline_z = global_min_z - 10  # Set baseline slightly below the minimum Z value

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Prepare color map
        cmap = plt.cm.get_cmap('viridis', num_sessions)

        # Plot lines for each session
        for i in range(num_sessions):
            xi = X[i, :]
            yi = Y[i, :]
            zi = Z[i, :]
            mask = ~np.isnan(zi)  # Mask to skip NaN values
            xi = xi[mask]
            yi = yi[mask]
            zi = zi[mask]

            # Plot each line
            ax.plot(xi, yi, zi, '-o', color=cmap(i), label=f'Session {session_list[i]}')

            # Create a solid filled surface under each line with adjusted baseline
            verts = [list(zip(xi, yi, zi)) + list(zip(xi[::-1], yi[::-1], np.full_like(zi, baseline_z)))]
            poly = art3d.Poly3DCollection(verts, color=cmap(i), alpha=0.5)
            ax.add_collection3d(poly)

            # Label the points with their mean value
            for xi_i, yi_i, zi_i in zip(xi, yi, zi):
                ax.text(xi_i, yi_i, zi_i + 0.05, f'{zi_i:.2f}', color='black', ha='center')

        # Connect corresponding points between sessions
        max_files = X.shape[1]
        for j in range(max_files):
            xi = X[:, j]
            yi = Y[:, j]
            zi = Z[:, j]
            mask = ~np.isnan(zi)
            xi = xi[mask]
            yi = yi[mask]
            zi = zi[mask]
            if len(xi) > 1:
                ax.plot(xi, yi, zi, '--', color='gray')  # Connect points between sessions

        ax.set_xlabel('File Index')
        ax.set_ylabel('Session')
        ax.set_zlabel('Mean Value')

        # Set y-ticks to session labels
        ax.set_yticks(range(num_sessions))
        ax.set_yticklabels(session_list)

        # Adjust Z-axis limits to fit data within the plot
        z_min = np.nanmin(Z)
        z_max = np.nanmax(Z)
        ax.set_zlim(baseline_z, z_max + abs(0.1 * z_max))

        # Automatically adjust subplot parameters to give more space
        plt.tight_layout()

        plt.title('Mean Values of .mat Files')
        plt.legend(loc='upper left')
        plt.show()

    def show_data_structure(self):
        """Displays the data structure of the dataset."""
        print("Data Structure")
        print("Root Folder:", self.root_dir)
        print("Available Dates (Selected):", self.subdates)
        print("Sessions in Selected Dates:", [s for _, s in self.sessions])
        # Uncomment below if you want to display the files
        # print("Files:", [f for _, _, f in self.files])
        
    def summarize_changes_and_plot(self, window_size=1000, save_fig=False, fig_dir='plots', palette='Set2'):
        """
        Summarizes the differences between each 'Pre' and 'Post' segment and plots a single boxplot.
        Individual data points are colored based on their session (hue).

        Parameters:
            window_size: int, the length of the 'Pre' and 'Post' segments for comparison.
            save_fig: bool, whether to save the figure to a file.
            fig_dir: str, the directory to save the plots if save_fig is True.
            palette: str, the color palette for the boxplot.
        """
        if save_fig and not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Initialize lists to store results
        pre_post_diff = []
        date_labels = []
        session_labels = []

        # Group the processed data by session (we include all dates)
        grouped_data = self.processed_data.groupby(['date', 'session'])

        for (date, session), group in grouped_data:
            # Sort the group by file name to ensure consistent ordering
            sorted_group = group.sort_values(by='file')
            sorted_keys = list(sorted_group[['date', 'session', 'file']].itertuples(index=False, name=None))
            num_changes = len(sorted_keys) - 1  # Number of changes in this session

            # Iterate over pairs of adjacent data entries within the same session
            for i in range(num_changes):
                key1 = sorted_keys[i]
                key2 = sorted_keys[i + 1]

                # Access filtered_data for the two consecutive entries
                data1 = self.processed_data[self.processed_data['date'] == key1[0]]
                data1 = data1[data1['session'] == key1[1]]
                data1 = data1[data1['file'] == key1[2]]['filtered_data'].values[0]

                data2 = self.processed_data[self.processed_data['date'] == key2[0]]
                data2 = data2[data2['session'] == key2[1]]
                data2 = data2[data2['file'] == key2[2]]['filtered_data'].values[0]

                if data1 is None or data2 is None:
                    continue

                # Use the window_size to extract 'Pre' and 'Post' segments
                pre_segment = data1[-window_size:]  # Last window_size points of data1
                post_segment = data2[:window_size]  # First window_size points of data2

                # Calculate the mean difference between 'Pre' and 'Post'
                mean_pre = np.mean(pre_segment)
                mean_post = np.mean(post_segment)
                diff = ((mean_post - mean_pre)/mean_pre) * 100

                # Store the results
                pre_post_diff.append(diff)
                date_labels.append(date)
                session_labels.append(session)

        
        # Create a DataFrame for plotting
        summary_data = pd.DataFrame({
            'Difference (Post - Pre)': pre_post_diff,
            'Date': date_labels,
            'Session': session_labels
        })
        summary_data['Combined'] = 'All Dates'
        summary_data.to_csv("summary_data.csv")
        # print(summary_data.head())
        # Ensure the 'Session' column is treated as a categorical variable
        summary_data['Session'] = summary_data['Session'].astype('category')
        # Create the boxplot
        plt.figure(figsize=(12, 6))
        
        # Create a single boxplot for all the differences
        sns.boxplot(x = 'Combined', y='Difference (Post - Pre)', data=summary_data, color='white', width=0.3)
        
        # Overlay individual points using swarmplot with hue for session (or date)
        swarm_plot = sns.swarmplot(x = 'Combined', y='Difference (Post - Pre)', data=summary_data, hue='Session', palette=palette, marker='o', size=7)

        # Manually set the legend
        # handles, labels = swarm_plot.get_legend_handles_labels()  # Get the handles and labels from swarmplot
        # print("!!!!!!!",labels)
        # plt.legend(handles, labels, title='Session', loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.legend(title='Session', loc='upper right', bbox_to_anchor=(1.25, 1))
        
        plt.title('Differences Between Post and Pre Segments with Session Hue')
        plt.xlabel('Difference (Post - Pre)')
        plt.ylabel('Percentage Change based on Pre(%)')
        # Save the plot if required
        if save_fig:
            plot_filename = f"summarized_pre_post_differences.png"
            plt.savefig(os.path.join(fig_dir, plot_filename))
            self.logger.info(f"Saved plot to {os.path.join(fig_dir, plot_filename)}")
            plt.close()
        else:
            plt.show()

    def process_baseline_data_for_violin(self,baseline_dir, window_size=1000, exclusion_dict=None):
        """
        Processes baseline data by extracting the last segment of one file as 'pre' and the first segment of the next file as 'post'.
        Returns a dictionary where the keys are dates and the values are dictionaries with pre and post-segment data from baseline.

        Parameters:
            baseline_dir: str, directory where the baseline data is stored.
            window_size: int, length of the pre and post segments.
            exclusion_dict: dict, any exclusions to apply when loading the data.

        Returns:
            dict, where the keys are the dates and values are dictionaries with pre and post-segment data from baseline.
        """
        baseline_segments = {}
        exclusion_dict = self.exclusion_dict or {}

        # Get list of dates in the baseline directory
        dates = [subdir for subdir in os.listdir(baseline_dir) if os.path.isdir(os.path.join(baseline_dir, subdir))]
        for date in dates:
            date_path = os.path.join(baseline_dir, date)

            # Exclude dates if specified
            if 'dates' in exclusion_dict and date in exclusion_dict['dates']:
                continue

            pre_segments = []
            post_segments = []

            # Get sessions for each date
            sessions = [subdir for subdir in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, subdir))]
            for session in sessions:
                session_path = os.path.join(date_path, session)

                # Exclude sessions if specified
                if 'sessions' in exclusion_dict and session in exclusion_dict['sessions'].get(date, []):
                    continue

                # Get .mat files for each session
                files = sorted([f for f in os.listdir(session_path) if f.endswith('.mat')])  # Sort files to ensure consistent order
                # Process pairs of adjacent files to extract pre and post segments
                num_files = len(files)
                for i in range(num_files - 1):  # Iterate over pairs of adjacent files
                    file1 = files[i]
                    file2 = files[i + 1]

                    file1_path = os.path.join(session_path, file1)
                    file2_path = os.path.join(session_path, file2)

                    try:
                        print("!!!!!!!!!!!!!!!!",date, session, file1, file2)
                        # Load data for the two consecutive files
                        data1 = scipy.io.loadmat(file1_path)
                        data2 = scipy.io.loadmat(file2_path)

                        baseline_env1 = data1['US_chunk'][0]  # Assuming baseline data has the same structure
                        baseline_env2 = data2['US_chunk'][0]

                        # Ensure both files have enough data for the window size
                        if len(baseline_env1) >= window_size and len(baseline_env2) >= window_size:
                            pre_segment = baseline_env1[-window_size:]  # Last window_size points of file1
                            post_segment = baseline_env2[:window_size]  # First window_size points of file2
                            envs = []
                            for i,tempt_env in enumerate([pre_segment, post_segment]):

                                tempt_env[tempt_env == 0] = np.nan  # Mark zeros as NaN
                                tempt_env = pd.Series(tempt_env).interpolate().fillna(method="bfill").fillna(method="ffill").values
                                z_scores = np.abs(stats.zscore(tempt_env))
                                tempt_env = np.where(z_scores > 2, np.nan, tempt_env)  # Mark outliers as NaN
                                tempt_env = pd.Series(tempt_env).interpolate().fillna(method="bfill").fillna(method="ffill").values
                                if date == "0307" and i==1:
                                    print("CLIP")
                                    tempt_env = np.clip(tempt_env, 500,700)
                                    plt.plot(tempt_env)
                                    plt.show()
                                envs.append(tempt_env)                      
                            print(np.max(envs[0]), np.min(envs[1]))
                            pre_segments.extend(envs[0])
                            post_segments.extend(envs[1])
                    except Exception as e:
                        print(f"Error loading {file1_path} or {file2_path}: {e}")

            if pre_segments and post_segments:
                baseline_segments[date] = {'pre': pre_segments, 'post': post_segments}  # Store pre and post segments for this date

        return baseline_segments


    def summarize_changes(self, window_size=1000, baseline_dir=None, save_fig=False, fig_dir='plots', palette='Set2',scale_data=False):
        """
        Summarizes the pre and post distributions for both the main data and baseline data, and creates a split view violin plot.

        Parameters:
            window_size: int, the length of the 'Pre' and 'Post' segments for comparison.
            baseline_dir: str, directory to pull baseline data from.
            save_fig: bool, whether to save the figure to a file.
            fig_dir: str, the directory to save the plots if save_fig is True.
            palette: str, the color palette for the violin plot.
        """
        if save_fig and not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Initialize lists to store results
        # main_pre_data = []
        # main_post_data = []
        main_data = []
        # baseline_pre_data = []
        # baseline_post_data = []
        baseline_data = []
        main_labels = []
        main_data_type_labels = []  # Pre or Post for main data
        main_source_labels = []  # Main source

        baseline_labels = []
        baseline_data_type_labels = []  # Pre or Post for baseline data
        baseline_source_labels = []  # Baseline source
        dates = []

        # Group the processed data by session (we include all dates)
        grouped_data = self.processed_data.groupby(['date', 'session'])
        
        for (date, session), group in grouped_data:
            dates.append(date)
            # Sort the group by file name to ensure consistent ordering
            sorted_group = group.sort_values(by='file')
            sorted_keys = list(sorted_group[['date', 'session', 'file']].itertuples(index=False, name=None))
            num_changes = len(sorted_keys) - 1  # Number of changes in this session

            # Iterate over pairs of adjacent data entries within the same session
            for i in range(num_changes):
                key1 = sorted_keys[i]
                key2 = sorted_keys[i + 1]

                # Access filtered_data for the two consecutive entries
                data1 = self.processed_data[self.processed_data['date'] == key1[0]]
                data1 = data1[data1['session'] == key1[1]]
                data1 = data1[data1['file'] == key1[2]]['filtered_data'].values[0]

                data2 = self.processed_data[self.processed_data['date'] == key2[0]]
                data2 = data2[data2['session'] == key2[1]]
                data2 = data2[data2['file'] == key2[2]]['filtered_data'].values[0]

                if data1 is None or data2 is None:
                    continue
                # Use the window_size to extract 'Pre' and 'Post' segments
                pre_segment = data1[-window_size:]  # Last window_size points of data1
                post_segment = data2[:window_size]  # First window_size points of data2   
                print(date, np.mean(pre_segment),np.mean(post_segment))  
                # plt.show()
                scaler = MinMaxScaler()
                if i==0:
                    print("Saved",date)
                    self.scaler[date].append(scaler)
                    print("RANGE!!!!:",min(pre_segment),max(post_segment))
                # Concatenate pre and post segments and flatten them
                tempt = np.concatenate([pre_segment, post_segment])
                # Reshape to (-1, 1) to treat each value as a feature
                segments_normalized = scaler.fit_transform(tempt.reshape(-1, 1))
                main_data.append(segments_normalized)
                # Label the data for main data
                main_labels.extend([f'{date}'] * (2 * window_size))
                main_data_type_labels.extend(['Pre'] * window_size + ['Post'] * window_size)
                main_source_labels.extend(['Main'] * (2 * window_size))

        dates = list(set(dates))  # Get unique dates

        # Process baseline data
        if baseline_dir is not None:
            baseline_segments = self.process_baseline_data_for_violin(baseline_dir, window_size=window_size)
            for date in dates:
                print("DATEDATE",date,type(date))
                if date in baseline_segments:
                    # print(len(baseline_pre_segment))
                    baseline_pre_segment = baseline_segments[date]['pre']
                    baseline_post_segment = baseline_segments[date]['post']
                    # plt.plot()
                    scaler = self.scaler[date][0]
                    # scaler = MinMaxScaler()
                    print(len(self.scaler[date]))
                    tempt = np.concatenate([baseline_pre_segment, baseline_post_segment])
                    if date == '0627':
                        # tempt = tempt -250
                        # mask = np.random.rand(*tempt.shape) < 0.9
                        # Replace elements where the mask is True with NaN
                        # tempt = np.where(mask, np.nan, tempt)
                        print("!!!!!!!!!!",tempt)
                    if date == '0729':
                        scaler = self.scaler[date][1]
                    # plt.hist(tempt.flatten(), bins=100)
                    # plt.show()
                    baseline = scaler.transform(tempt.reshape(-1, 1))
                    # baseline = scaler.fit_transform(tempt.reshape(-1, 1))
                    baseline_data.append(tempt)
                    # Label the baseline data
                    baseline_labels.extend([f'{date}'] * (2 * window_size))
                    baseline_data_type_labels.extend(['Pre'] * window_size + ['Post'] * window_size)
                    baseline_source_labels.extend(['Baseline'] * (2 * window_size))
                    print(date,np.mean(baseline_pre_segment), np.mean(baseline_post_segment))
                    

        # combined_values = np.concatenate([main_data,baseline_data]).flatten()
        # combined_segments = main_data_type_labels + baseline_data_type_labels
        # combined_sources = main_source_labels + baseline_source_labels
        # combined_labels = main_labels + baseline_labels
        # plot_data = pd.DataFrame({
        #     'Value':  combined_values,
        #     'Segment': combined_segments,
        #     'Source': combined_sources,
        #     'Date': combined_labels
        # })
        baseline_plot = pd.DataFrame({
            'Value': np.concatenate(baseline_data).flatten(),
            'Segment': baseline_data_type_labels,
            'Date': baseline_labels
        })
        main_plot = pd.DataFrame({
            'Value': np.concatenate(main_data).flatten(),
            'Segment': main_data_type_labels,
            'Date': main_labels
        })
        main_plot.to_csv("mainplot.csv")
        if scale_data:
            # Plotting using violin plots
            plt.figure(figsize=(14, 8))
            sns.violinplot(x='Date', y='Value', hue='Segment', data=main_plot, split=True, palette=palette)
            plt.title('Pre and Post Distributions: Scaled Data')
            plt.xlabel('Date')
            plt.ylabel('Scaled Values')
            plt.legend()
        else:
            # g = sns.FacetGrid(plot_data, row='Date', hue='Segment', aspect=15, height=0.5, palette=palette, sharex=False)

            # # Map the kdeplots
            # g.map(sns.kdeplot, 'Value',
            #     bw_adjust=1, clip_on=False,
            #     fill=True, alpha=0.6, linewidth=1.5)

            # # Add a horizontal line at the bottom of each plot
            # g.map(plt.axhline, y=0, lw=2, clip_on=False)

            # # Set the labels and titles
            # g.set_titles("")
            # g.set(yticks=[])
            # g.despine(bottom=True, left=True)

            # # Adjust the subplots to overlap
            # g.fig.subplots_adjust(hspace=-0.7)

            # # Add a legend
            # g.add_legend(title='Segment')

            # plt.xlabel('Value')
            # plt.show()
            # Prepare the data
            unique_dates = sorted(plot_data['Date'].unique())
            n_dates = len(unique_dates)
            colors = sns.color_palette(palette, n_colors=2)  # For 'Pre' and 'Post'

            # Set up the figure
            fig, axes = plt.subplots(n_dates, 1, figsize=(10, n_dates * 1.5), sharex=True)
            if n_dates == 1:
                axes = [axes]  # Ensure axes is iterable

            for ax, date in zip(axes, unique_dates):
                data = plot_data[plot_data['Date'] == date]
                # Plot 'Pre' distribution
                sns.kdeplot(data=data[data['Segment'] == 'Pre'], x='Value', ax=ax, fill=True, color=colors[0], alpha=0.6, label='Pre')
                # Plot 'Post' distribution
                sns.kdeplot(data=data[data['Segment'] == 'Post'], x='Value', ax=ax, fill=True, color=colors[1], alpha=0.6, label='Post')
                ax.set_ylabel(date)
                ax.set_yticks([])
                ax.legend(loc='upper right')
                ax.set_xlabel('')

            # Adjust the spacing between subplots to create the ridgeline effect
            plt.subplots_adjust(hspace=-0.5)
            plt.xlabel('Value')
            plt.suptitle('Pre and Post Distributions: Absolute Data Values')
            plt.show()

        
        return baseline_plot
        # # plot_data.to_csv("plot_data.csv")

        # plt.figure(figsize=(14, 8))
        # plot_data.replace(0, np.nan, inplace=True)
        # baseline_plot.replace(0, np.nan, inplace=True)
        # plot_data.to_csv("plot_data.csv")
        # # sns.violinplot(x='Date', y='Value', hue='Segment', split = True, data=baseline_data, palette=palette)
        # # sns.violinplot(x='Date', y='Value', hue='Segment', split = True, data=baseline_plot, color='lightgrey')
        # sns.violinplot(x='Date', y='Value', hue='Segment', split = True, data=main_plot, color='darkblue')
        
        # # plt.title('Pre and Post Distributions: Main vs. Baseline Data')
        # # plt.xlabel('Date')
        # # plt.ylabel('Values')
        # plt.legend()

        # # Save the plot if required
        # if save_fig:
        #     plot_filename = f"summarized_changes_split_violin_plot.png"
        #     plt.savefig(os.path.join(fig_dir, plot_filename))
        #     self.logger.info(f"Saved plot to {os.path.join(fig_dir, plot_filename)}")
        #     plt.close()
        # else:
        #     plt.show()
        
        # return baseline_plot