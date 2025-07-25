import numpy as np
import pandas as pd
import re
import os
import logging
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import entropy
from datetime import datetime
from tqdm import tqdm


# Function to extract turn counts from a coded turn string
def extract_turn_counts(turn_string):
    pattern = r'(\d)'
    matches = re.findall(pattern, turn_string)
    return Counter(matches)

def mean_absolute_change(series):
    return np.mean(np.abs(np.diff(series)))

def clinician_to_participant_ratio(group):
    speaker_turns = group.groupby('speaker')['turns'].sum()
    clinician_turns = speaker_turns.get('0', 0)
    participant_turns = speaker_turns.drop('0', errors='ignore').sum()
    return clinician_turns / participant_turns if participant_turns > 0 else np.nan

def compute_speaker_level(df):
    agg_dict = {
        'turns': 'sum',
    }

    if 'session' in df.columns:
        agg_dict['session'] = pd.Series.nunique
    if 'bin' in df.columns:
        agg_dict['bin'] = 'count'

    speaker_level = (
        df.groupby(['group', 'speaker'], as_index=False)
        .agg(agg_dict)
        .rename(columns={
            'turns': 'total_turns',
            'session': 'unique_sessions' if 'session' in df.columns else None,
            'bin': 'bins_appeared_in' if 'bin' in df.columns else None
        })
    )

    return speaker_level.loc[:, ~speaker_level.columns.str.match(r'^None$')]
    
def compute_group_level(df):
    group_agg = {
        'turns': 'sum',
        'speaker': pd.Series.nunique,
    }

    if 'bin' in df.columns:
        group_agg['bin'] = 'count'

    if 'session' in df.columns:
        group_agg['session'] = pd.Series.nunique

    group_level = (
        df.groupby('group', as_index=False)
        .agg(group_agg)
        .rename(columns={
            'turns': 'total_turns',
            'session': 'num_sessions',
            'speaker': 'num_participants',
            'bin': 'bins_covered' if 'bin' in df.columns else None
        })
    )

    # Drop columns with meaningless names if they exist
    return group_level.loc[:, ~group_level.columns.str.match(r'^None$')]

def compute_bin_level(df, grouping_cols):
    bin_level = df.copy()
    bin_totals = df.groupby(grouping_cols)['turns'].transform('sum')
    bin_level['proportion_of_bin_turns'] = bin_level['turns'] / bin_totals
    return bin_level

def compute_session_level(turn_totals):
    session_summary = (
        turn_totals.groupby(['session', 'group'])
        .agg(total_turns=('turns', 'sum'))
        .reset_index()
    )

    entropy_data = (
        turn_totals.groupby(['session', 'group', 'speaker'])['turns'].sum()
        .groupby(['session', 'group'])
        .apply(lambda x: entropy(x))
        .reset_index(name='turn_entropy')
    )

    ratios = (
        turn_totals.groupby(['session', 'group'])
        .apply(clinician_to_participant_ratio)
        .reset_index(name='clinician_participant_ratio')
    )

    session_summary = session_summary.merge(entropy_data, on=['session', 'group'], how='left')
    session_summary = session_summary.merge(ratios, on=['session', 'group'], how='left')

    return session_summary

def compute_participation_level(turn_totals, has_bin=False):
    
    participation_level = (
        turn_totals.groupby(['group', 'session', 'speaker'], as_index=False)
        .agg({'turns': 'sum'})
    )
    session_totals = participation_level.groupby(['session', 'group'])['turns'].transform('sum')
    participation_level['proportion_of_session_turns'] = participation_level['turns'] / session_totals

    # Bin stats per participant-session
    if has_bin:
        bin_stats = (
            turn_totals.groupby(['session', 'group', 'speaker'])
            .agg(
                mean_turns=('turns', 'mean'),
                std_turns=('turns', 'std'),
                var_turns=('turns', 'var'),
                min_turns=('turns', 'min'),
                max_turns=('turns', 'max'),
            )
            .reset_index()
        )
        bin_stats['cv_turns'] = bin_stats['std_turns'] / bin_stats['mean_turns']

        changes = (
            turn_totals.sort_values(['session', 'group', 'speaker', 'bin'])
            .groupby(['session', 'group', 'speaker'])['turns']
            .agg(avg_change_turns=mean_absolute_change)
            .reset_index()
        )

        participation_level = participation_level.merge(bin_stats, on=['session', 'group', 'speaker'], how='left')
        participation_level = participation_level.merge(changes, on=['session', 'group', 'speaker'], how='left')
    
    return participation_level

# --- Transition Matrices and Ratios ---
def extract_sequence(turn_string):
    return re.findall(r'\d', turn_string)

def build_transition_matrix(sequences):
    transition_counts = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_speaker = seq[i]
            to_speaker = seq[i + 1]
            transition_counts[from_speaker][to_speaker] += 1

    # Flatten speaker list and sort
    speakers = sorted(set(transition_counts) | {k for d in transition_counts.values() for k in d})
    matrix = pd.DataFrame(0, index=speakers, columns=speakers, dtype=int)

    for from_spk, to_dict in transition_counts.items():
        for to_spk, count in to_dict.items():
            matrix.loc[from_spk, to_spk] = count

    return matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

def compute_transition_metrics(df):
    """
    For each group, compute a transition matrix based on the 'turns' strings,
    and derive clinician/participant turn-taking ratios.
    Returns:
        {
            'transition_matrices': {group_id: DataFrame},
            'speaker_ratios': list of dicts
        }
    """
    speaker_matrices = {}
    speaker_ratios = []

    for group, group_df in df.groupby('group'):
        # Ensure valid sequences
        sequences = [extract_sequence(ts) for ts in group_df['turns'] if isinstance(ts, str) and ts.strip()]
        if not sequences:
            continue

        matrix = build_transition_matrix(sequences)
        speaker_matrices[str(group)] = matrix

        # Compute ratios
        speakers = matrix.columns.astype(str)
        participants = [s for s in speakers if s != '0']
        ptp = matrix.loc[participants, participants].to_numpy().sum() if participants else np.nan
        ptc = matrix.loc[participants, '0'].sum() if '0' in matrix.columns else np.nan
        cpp = matrix.loc['0', participants].sum() if '0' in matrix.index else np.nan

        speaker_ratios.append({
            'group': group,
            'participant_to_participant': ptp,
            'participant_to_clinician': ptc,
            'clinician_to_participant': cpp
        })

    return {
        'transition_matrices': speaker_matrices,
        'speaker_ratios': pd.DataFrame(speaker_ratios)
    }

def _analyze_convo_turns_file(df):
    required_cols = ['group', 'turns']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    has_session = 'session' in df.columns
    has_bin = 'bin' in df.columns

    grouping_cols = ['group', 'speaker']
    if has_session:
        grouping_cols.append('session')
    if has_bin:
        grouping_cols.append('bin')

    # Extract turn data
    rows = []
    for _, row in df.iterrows():
        if pd.isna(row['turns']):
            continue
        turn_counts = extract_turn_counts(row['turns'])
        for speaker, count in turn_counts.items():
            rows.append({
                'group': row['group'],
                'session': row.get('session'),
                'speaker': speaker,
                'bin': row.get('bin'),
                'turns': count,
            })

    turn_totals = pd.DataFrame(rows)

    # Compute metrics
    ct_data = {
        'speaker_level': compute_speaker_level(turn_totals),
        'group_level': compute_group_level(turn_totals),
    }

    if has_bin:
        bin_grouping = ['group']
        if has_session:
            bin_grouping.append('session')
        bin_grouping.append('bin')  # always include 'bin'
        turn_totals = compute_bin_level(turn_totals, grouping_cols=bin_grouping)
        ct_data['bin_level'] = turn_totals

    if has_session:
        ct_data['session_level'] = compute_session_level(turn_totals)
        ct_data['participation_level'] = compute_participation_level(turn_totals, has_bin=has_bin)
    
    ct_data.update(compute_transition_metrics(df))

    return ct_data

# Summary statistics with coefficient of variation
def summarize(df, level_name):
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.agg(['mean', 'std', 'min', 'max']).transpose()
    summary['cv'] = summary['std'] / summary['mean']
    summary.reset_index(inplace=True)
    summary.columns = ['metric', 'mean', 'std', 'min', 'max', 'cv']
    summary.insert(0, 'level', level_name)
    return summary

def write_if_not_empty(df, writer, sheet_name):
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

def analyze_digital_convo_turns(input_dir, output_dir, test=False):
    digital_convo_turns_dir = os.path.join(output_dir, 'digital_convo_turns')
    try:
        os.makedirs(digital_convo_turns_dir, exist_ok=True)
        logging.info(f"Created directory: {digital_convo_turns_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {digital_convo_turns_dir}: {e}")
        return

    ct_files = [f for f in Path(input_dir).rglob('*_ConvoTurns.xlsx')]
    logging.info(f"Found {len(ct_files)} files in {input_dir}.")
    results = []

    for ct_file in tqdm(ct_files, desc="Analyzing conversation turns"):
        try:
            xls = pd.ExcelFile(ct_file)
            if not xls.sheet_names:
                logging.warning(f"No sheets found in file: {ct_file.name}")
                continue
            df = xls.parse(xls.sheet_names[0])
            if df.empty:
                logging.warning(f"Empty data in file: {ct_file.name}")
                continue

            ct_data = _analyze_convo_turns_file(df)

            # Extract all data levels (with fallback to empty df)
            bin_level = ct_data.get('bin_level', pd.DataFrame())
            participation_level = ct_data.get('participation_level', pd.DataFrame())
            session_level = ct_data.get('session_level', pd.DataFrame())
            speaker_level = ct_data.get('speaker_level', pd.DataFrame())
            group_level = ct_data.get('group_level', pd.DataFrame())
            speaker_ratios = ct_data.get('speaker_ratios', pd.DataFrame())
            speaker_matrices = ct_data.get('transition_matrices', {})

            # Compile summary stats safely
            summary_levels = {
                'session': session_level,
                'participation': participation_level,
                'speaker': speaker_level,
                'group': group_level,
            }

            summary_frames = []
            for level_name, df_level in summary_levels.items():
                if isinstance(df_level, pd.DataFrame) and not df_level.empty:
                    try:
                        summary_frames.append(summarize(df_level, level_name))
                    except Exception as e:
                        logging.warning(f"Could not summarize {level_name} level: {e}")

            summary_stats_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

            # Save to Excel
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            base_name = ct_file.stem.replace('_ConvoTurns', '')
            final_path = os.path.join(digital_convo_turns_dir, f"{base_name}_ConvoTurnAnalysis_{timestamp}.xlsx")

            with pd.ExcelWriter(final_path, engine='xlsxwriter') as writer:
                write_if_not_empty(bin_level, writer, 'Bin_Level_Turns')
                write_if_not_empty(participation_level, writer, 'Participation_Level_Turns')
                write_if_not_empty(session_level, writer, 'Session_Level_Summary')
                write_if_not_empty(speaker_level, writer, 'Speaker_Level_Turns')
                write_if_not_empty(group_level, writer, 'Group_Level_Summary')
                write_if_not_empty(summary_stats_all, writer, 'Summary_Statistics')
                write_if_not_empty(speaker_ratios, writer, 'Speaker_Level_Ratios')

                for name, matrix in speaker_matrices.items():
                    sheet_name = f"Trans_Speaker_{name}"[:31]  # Excel max sheet name = 31
                    matrix.to_excel(writer, sheet_name=sheet_name)

            if test:
                results.append(ct_data)

        except Exception as e:
            logging.error(f"Unexpected error with file {ct_file.name}: {e}")

    if test:
        return results
