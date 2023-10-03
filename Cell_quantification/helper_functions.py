# libraries
import pandas as pd

# Helper functions


#  To perform cell ('Class') count for a given slide
def cell_count(file, area):
    selected = file[file['Name'] == area]
    counts = selected['Class'].value_counts()
    return counts


# Quantify cell counts for a list of files
def calculate_cell_count_df(file_path,
                            prediction_list,
                            area  # e.g. 'Grey_matter'
                            ):

    # Quantify cell density
    image_list = []
    n_astro = []
    n_neuron = []
    n_oligo = []
    n_others = []
    n_ambiguous = []

    for i in (prediction_list):

        # read in file
        p = pd.read_csv(file_path + i, sep='\t')

        # Extract information
        image_name = p['Image'][0]

        # Cell counts in GM
        gm_counts = cell_count(p,
                               area)

        # combined extracted info
        image_list.append(image_name)
        n_astro.append(gm_counts['Astro'])
        n_neuron.append(gm_counts['Neuron'])
        n_oligo.append(gm_counts['Oligo'])
        n_others.append(gm_counts['Others'])
        n_ambiguous.append(gm_counts['Ambiguous'])

    cell_count_df = pd.DataFrame({'Image_name': image_list,
                                  'Astro': n_astro,
                                  'Neuron': n_neuron,
                                  'Oligo': n_oligo,
                                  'Others': n_others,
                                  'Ambiguous': n_ambiguous})
    # Add extra metrics
    cell_count_df_output = cell_count_df.copy()
    # Total cell counts (excluding Ambiguous)
    cell_count_df_output.loc[:, 'Total'] = cell_count_df_output['Astro'] + cell_count_df_output['Neuron'] + cell_count_df_output['Oligo'] + cell_count_df_output['Others']

    # Calculate cell proportion
    cell_count_df_output.loc[:, 'p_Astro'] = cell_count_df_output['Astro']/cell_count_df_output['Total']
    cell_count_df_output.loc[:, 'p_Neuron'] = cell_count_df_output['Neuron']/cell_count_df_output['Total']
    cell_count_df_output.loc[:, 'p_Oligo'] = cell_count_df_output['Oligo']/cell_count_df_output['Total']
    cell_count_df_output.loc[:, 'p_Others'] = cell_count_df_output['Others']/cell_count_df_output['Total']
    return cell_count_df_output


# Quantify cell counts for a list of files (for STN which some BG slides may not have)
def calculate_cell_count_df_missing(file_path,
                            prediction_list,
                            area  # e.g. 'Grey_matter'
                            ):

    # Quantify cell density
    image_list = []
    n_astro = []
    n_neuron = []
    n_oligo = []
    n_others = []
    n_ambiguous = []
    faulty = []

    for i in (prediction_list):

        # read in file
        p = pd.read_csv(file_path + i, sep='\t')

        # Extract information
        image_name = p['Image'][0]

        # Make sure 'area' exists in the prediction file
        x = list(set(p['Name']))
        if ((area in x)==False):
            faulty.append(image_name)
            continue

        # Cell counts in GM
        gm_counts = cell_count(p,
                               area)

        # combined extracted info
        image_list.append(image_name)
        n_astro.append(gm_counts['Astro'])
        n_neuron.append(gm_counts['Neuron'])
        n_oligo.append(gm_counts['Oligo'])
        n_others.append(gm_counts['Others'])
        n_ambiguous.append(gm_counts['Ambiguous'])

    cell_count_df = pd.DataFrame({'Image_name': image_list,
                                  'Astro': n_astro,
                                  'Neuron': n_neuron,
                                  'Oligo': n_oligo,
                                  'Others': n_others,
                                  'Ambiguous': n_ambiguous})
    # Add extra metrics
    cell_count_df_output = cell_count_df.copy()
    # Total cell counts (excluding Ambiguous)
    cell_count_df_output.loc[:, 'Total'] = cell_count_df_output['Astro'] + cell_count_df_output['Neuron'] + cell_count_df_output['Oligo'] + cell_count_df_output['Others']

    # Calculate cell proportion
    cell_count_df_output.loc[:, 'p_Astro'] = cell_count_df_output['Astro']/cell_count_df_output['Total']
    cell_count_df_output.loc[:, 'p_Neuron'] = cell_count_df_output['Neuron']/cell_count_df_output['Total']
    cell_count_df_output.loc[:, 'p_Oligo'] = cell_count_df_output['Oligo']/cell_count_df_output['Total']
    cell_count_df_output.loc[:, 'p_Others'] = cell_count_df_output['Others']/cell_count_df_output['Total']
    return cell_count_df_output, faulty


# Quantify cell counts for a list of files
def calculate_tau_positive_cell_count_df(file_path,
                                         prediction_list,
                                         tau_positive_threshold,
                                         dab_feature,
                                         area  # e.g. 'Grey_matter'
                                         ):

    # Quantify cell density
    image_list = []
    n_astro = []
    n_neuron = []
    n_oligo = []
    n_others = []
    n_ambiguous = []

    for i in (prediction_list):

        # read in file
        p = pd.read_csv(file_path + i, sep='\t')

        # Extract information
        image_name = p['Image'][0]

        # Subset only tau positive cells**
        p_tau_pos = p[p[dab_feature] >= tau_positive_threshold]

        # tau positive counts in GM
        gm_counts = cell_count(p_tau_pos,
                               area)

        # combined extracted info
        image_list.append(image_name)
        n_astro.append(gm_counts['Astro'])
        n_neuron.append(gm_counts['Neuron'])
        n_oligo.append(gm_counts['Oligo'])
        n_others.append(gm_counts['Others'])
        n_ambiguous.append(gm_counts['Ambiguous'])

    tau_positive_count_df = pd.DataFrame({'Image_name': image_list,
                                          'Astro+': n_astro,
                                          'Neuron+': n_neuron,
                                          'Oligo+': n_oligo,
                                          'Others+': n_others,
                                          'Ambiguous+': n_ambiguous})
    return tau_positive_count_df


# Quantify cell counts for a list of files
def calculate_tau_positive_cell_count_df_missing(file_path,
                                         prediction_list,
                                         tau_positive_threshold,
                                         dab_feature,
                                         area  # e.g. 'Grey_matter'
                                         ):

    # Quantify cell density
    image_list = []
    n_astro = []
    n_neuron = []
    n_oligo = []
    n_others = []
    n_ambiguous = []
    faulty = []

    for i in (prediction_list):

        # read in file
        p = pd.read_csv(file_path + i, sep='\t')

        # Extract information
        image_name = p['Image'][0]

        # Make sure 'area' exists in the prediction file
        x = list(set(p['Name']))
        if ((area in x)==False):
            faulty.append(image_name)
            continue

        # Subset only tau positive cells**
        p_tau_pos = p[p[dab_feature] >= tau_positive_threshold]

        # tau positive counts in GM
        gm_counts = cell_count(p_tau_pos,
                               area)

        # combined extracted info
        image_list.append(image_name)
        n_astro.append(gm_counts['Astro'])
        n_neuron.append(gm_counts['Neuron'])
        n_oligo.append(gm_counts['Oligo'])
        n_others.append(gm_counts['Others'])
        n_ambiguous.append(gm_counts['Ambiguous'])

    tau_positive_count_df = pd.DataFrame({'Image_name': image_list,
                                          'Astro+': n_astro,
                                          'Neuron+': n_neuron,
                                          'Oligo+': n_oligo,
                                          'Others+': n_others,
                                          'Ambiguous+': n_ambiguous})
    return tau_positive_count_df, faulty


# Quantify cell counts for a list of files
def calculate_tau_negative_cell_count_df_missing(file_path,
                                         prediction_list,
                                         tau_positive_threshold,
                                         dab_feature,
                                         area  # e.g. 'Grey_matter'
                                         ):

    # Quantify cell density
    image_list = []
    n_astro = []
    n_neuron = []
    n_oligo = []
    n_others = []
    n_ambiguous = []
    faulty = []

    for i in (prediction_list):

        # read in file
        p = pd.read_csv(file_path + i, sep='\t')

        # Extract information
        image_name = p['Image'][0]

        # Make sure 'area' exists in the prediction file
        x = list(set(p['Name']))
        if ((area in x)==False):
            faulty.append(image_name)
            continue

        # Subset only tau positive cells**
        p_tau_neg = p[p[dab_feature] < tau_positive_threshold]

        # tau negative counts in GM
        gm_counts = cell_count(p_tau_neg,
                               area)

        # combined extracted info
        image_list.append(image_name)
        n_astro.append(gm_counts['Astro'])
        n_neuron.append(gm_counts['Neuron'])
        n_oligo.append(gm_counts['Oligo'])
        n_others.append(gm_counts['Others'])
        n_ambiguous.append(gm_counts['Ambiguous'])

    tau_negative_count_df = pd.DataFrame({'Image_name': image_list,
                                          'Astro-': n_astro,
                                          'Neuron-': n_neuron,
                                          'Oligo-': n_oligo,
                                          'Others-': n_others,
                                          'Ambiguous-': n_ambiguous})
    return tau_negative_count_df, faulty


#  To perform cell density quantification
def cell_density(df):
    df_output = df.copy()
    df_output.loc[:, 'd_Astro'] = df_output['Astro']/df_output['Area']
    df_output.loc[:, 'd_Oligo'] = df_output['Oligo']/df_output['Area']
    df_output.loc[:, 'd_Neuron'] = df_output['Neuron']/df_output['Area']
    df_output.loc[:, 'd_Others'] = df_output['Others']/df_output['Area']

    df_output.loc[:, 'keycell'] = df_output['Astro'] + df_output['Oligo'] + df_output['Neuron']
    df_output.loc[:, 'd_keycell'] = df_output['keycell']/df_output['Area']

    df_output.loc[:, 'd_Astro+'] = df_output['Astro+']/df_output['Area']
    df_output.loc[:, 'd_Oligo+'] = df_output['Oligo+']/df_output['Area']
    df_output.loc[:, 'd_Neuron+'] = df_output['Neuron+']/df_output['Area']
    df_output.loc[:, 'd_Others+'] = df_output['Others+']/df_output['Area']

    df_output.loc[:, 'keycell+'] = df_output['Astro+'] + df_output['Oligo+'] + df_output['Neuron+']
    df_output.loc[:, 'd_keycell+'] = df_output['keycell+']/df_output['Area']

    # Calculate GNR
    df_output.loc[:, 'GNR'] = (df_output['Astro']+df_output['Oligo'])/df_output['Neuron']
    return df_output

#  To perform cell density quantification
def cell_density_negative(df):
    df_output = df.copy()
    df_output.loc[:, 'd_Astro'] = df_output['Astro']/df_output['Area']
    df_output.loc[:, 'd_Oligo'] = df_output['Oligo']/df_output['Area']
    df_output.loc[:, 'd_Neuron'] = df_output['Neuron']/df_output['Area']
    df_output.loc[:, 'd_Others'] = df_output['Others']/df_output['Area']

    df_output.loc[:, 'keycell'] = df_output['Astro'] + df_output['Oligo'] + df_output['Neuron']
    df_output.loc[:, 'd_keycell'] = df_output['keycell']/df_output['Area']

    df_output.loc[:, 'd_Astro-'] = df_output['Astro-']/df_output['Area']
    df_output.loc[:, 'd_Oligo-'] = df_output['Oligo-']/df_output['Area']
    df_output.loc[:, 'd_Neuron-'] = df_output['Neuron-']/df_output['Area']
    df_output.loc[:, 'd_Others-'] = df_output['Others-']/df_output['Area']

    df_output.loc[:, 'keycell-'] = df_output['Astro-'] + df_output['Oligo-'] + df_output['Neuron-']
    df_output.loc[:, 'd_keycell-'] = df_output['keycell-']/df_output['Area']

    # Calculate GNR
    df_output.loc[:, 'GNR-'] = (df_output['Astro-']+df_output['Oligo-'])/df_output['Neuron-']
    df_output.loc[:, 'GNR'] = (df_output['Astro']+df_output['Oligo'])/df_output['Neuron']
    return df_output


#  To perform extra cell proportion calculation
def cell_proportion_extra(df):
    df_output = df.copy()

    # cell proportion
    df_output.loc[:, 'keycell'] = df_output['Astro'] + df_output['Oligo'] + df_output['Neuron']
    df_output.loc[:, 'pk_Astro'] = df_output['Astro']/df_output['keycell']
    df_output.loc[:, 'pk_Oligo'] = df_output['Oligo']/df_output['keycell']
    df_output.loc[:, 'pk_Neuron'] = df_output['Neuron']/df_output['keycell']
    df_output.loc[:, 'p_keycell'] = df_output['keycell']/df_output['Total']

    # Out of tau positive cells, calculate proportion of each cell type
    df_output.loc[:, 'keycell+'] = df_output['Astro+'] +df_output['Oligo+']+df_output['Neuron+']
    df_output.loc[:, 'p_keycell+'] = df_output['keycell+']/df_output['Total']
    df_output.loc[:, 'pk_Astro+'] = df_output['Astro+']/df_output['keycell']
    df_output.loc[:, 'pk_Oligo+'] = df_output['Oligo+']/df_output['keycell']
    df_output.loc[:, 'pk_Neuron+'] = df_output['Neuron+']/df_output['keycell']

    # Calculate GNR
    df_output.loc[:, 'GNR'] = (df_output['Astro']+df_output['Oligo'])/df_output['Neuron']
    return df_output

#  To perform extra cell proportion calculation
def cell_proportion_extra_negative(df):
    df_output = df.copy()

    # cell proportion
    df_output.loc[:, 'keycell'] = df_output['Astro'] + df_output['Oligo'] + df_output['Neuron']
    df_output.loc[:, 'pk_Astro'] = df_output['Astro']/df_output['keycell']
    df_output.loc[:, 'pk_Oligo'] = df_output['Oligo']/df_output['keycell']
    df_output.loc[:, 'pk_Neuron'] = df_output['Neuron']/df_output['keycell']
    df_output.loc[:, 'p_keycell'] = df_output['keycell']/df_output['Total']

    # Out of tau positive cells, calculate proportion of each cell type
    df_output.loc[:, 'keycell-'] = df_output['Astro-'] +df_output['Oligo-']+df_output['Neuron-']
    df_output.loc[:, 'p_keycell-'] = df_output['keycell-']/df_output['Total']
    df_output.loc[:, 'pk_Astro-'] = df_output['Astro-']/df_output['keycell']
    df_output.loc[:, 'pk_Oligo-'] = df_output['Oligo-']/df_output['keycell']
    df_output.loc[:, 'pk_Neuron-'] = df_output['Neuron-']/df_output['keycell']

    # Calculate GNR
    df_output.loc[:, 'GNR'] = (df_output['Astro-']+df_output['Oligo-'])/df_output['Neuron-']
    return df_output


def data_inspection(dat):
    """
    Creates 3 tables summary of:
    1. Slides / region / stage
    2. No. of subjects / region
    3. No. of subjects / PSP subtype
    Prints No. of unique subjects
    """

    stages = list(set(dat['Stage_SK']))
    regions = list(set(dat['region_name']))

    s_r = []
    # Create table 1)
    for s in stages:  # for each PSP stage
        s_dat = dat[dat['Stage_SK'] == s]
        for_all_regions = []

        for r in regions:  # for each region
            # get number of slides
            r_ = s_dat[s_dat['region_name'] == r].shape[0]
            for_all_regions.append(r_)

        s_r.append(for_all_regions)

    output1 = pd.DataFrame(data={
                                'Stage 2': s_r[0],
                                'Stage 3': s_r[1],
                                'Stage 4': s_r[2],
                                'Stage 5': s_r[3],
                                'Stage 6': s_r[4]
                                })
    output1.insert(0, 'Regions', regions)

    # Creates table 2)
    PSP_subs = dat[['Patient_ID', 'Stage_SK']]
    PSP_subs = PSP_subs.drop_duplicates(subset=['Patient_ID'])
    PSP_sk = PSP_subs['Stage_SK'].value_counts()
    output2_ = PSP_sk.rename_axis('Stage_SK').reset_index(name='Counts')
    output2 = output2_.sort_values(by='Stage_SK')

    # Creates table 3)
    PSP_subs = dat[['Patient_ID', 'MDS-PSP last visit']]
    PSP_subs = PSP_subs.drop_duplicates(subset=['Patient_ID'])
    PSP_subtypes = PSP_subs['MDS-PSP last visit'].value_counts()
    output3 = PSP_subtypes.rename_axis('PSP subtype').reset_index(name='Counts')

    print('No. of unique patients: ', len(list(set(dat['Patient_ID']))))
    return output1, output2, output3


def correlation_table(dat, stage, tau_types):
    """
    Calculates correlation table
    between tau burden & stage rating e.g. Stage_SK.
    """
    # tau_types = ['d_keycell+',
    #              'Tau_keycell_density',
    #              'CB_density',
    #              'NFT_density',
    #              'TA_density',
    #              'Others_density',
    #              'Others_AF'
    #              ]

    corr_p = [spearmanr(dat[[i, stage]]) for i in tau_types]   # get both r, p
    r_val = [round(i[0], 3) for i in corr_p]  # separates r out
    p_val = [i[1] for i in corr_p]  # separates p out
    # [round(spearmanr(dat[[i, stage]])[0], 3) for i in tau_types]
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                stage: r_val,
                                'p_val': p_val
                                })
    return output


def correlation_region_tau_stage(dat, stage):
    """
    Calculates correlation between region-specific tau burden
    for each burden type & stage rating of choice e.g. Stage_SK.
    To see whcih region & tau burden type correlates best with Stage_SK.
    """
    kovacs_regions = ['Globus pallidus',
                      'Subthalamic nucleus',
                      'Striatum',
                      'Pre-frontal',
                      'Dentate nucleus',
                      'Occipital'
                      ]

    tau_types = ['Total_tau_density',
                 'Tau_hallmark_density',
                 'CB_density',
                 'NFT_density',
                 'TA_density',
                 'Others_density',
                 'Others_AF'
                ]
    regional_corr = []
    regional_pval = []
    # for each region, calculates corr with all tau types
    for r in kovacs_regions:
        r_dat = dat[dat['region_name'] == r]
        r_corr = [round(spearmanr(r_dat[[i, stage]])[0], 3) for i in tau_types]
        p_val = [round(spearmanr(r_dat[[i, stage]])[1], 3) for i in tau_types]
        regional_corr.append(r_corr)
        regional_pval.append(p_val)
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                kovacs_regions[0]: regional_corr[0],
                                kovacs_regions[0]+'_pval': regional_pval[0],
                                kovacs_regions[1]: regional_corr[1],
                                kovacs_regions[1]+'_pval': regional_pval[1],
                                kovacs_regions[2]: regional_corr[2],
                                kovacs_regions[2]+'_pval': regional_pval[2],
                                kovacs_regions[3]: regional_corr[3],
                                kovacs_regions[3]+'_pval': regional_pval[3],
                                kovacs_regions[4]: regional_corr[4],
                                kovacs_regions[4]+'_pval': regional_pval[4],
                                kovacs_regions[5]: regional_corr[5],
                                kovacs_regions[5]+'_pval': regional_pval[5]
                                })
    print('Correlation between region-specific tau & '+stage)
    return output


def correlation_region_tau_region_stage(dat):
    """
    Calculates correlation between region-specific tau burden
    for each burden type & region-specific severity rating.
    E.g. (tau in dentate nucleus & DE_SK)
    """
    kovacs_regions_rating = {'Globus pallidus': 'GP_SK',
                             'Subthalamic nucleus': 'STN_SK',
                             'Striatum': 'STR_SK',
                             'Pre-frontal': 'FCF_SK',
                             'Dentate nucleus': 'DE_SK',
                             'Occipital': 'OC_SK'
                             }

    tau_types = ['Total_tau_density',
                 'Tau_hallmark_density',
                 'CB_density',
                 'NFT_density',
                 'TA_density',
                 'Others_density',
                 'Others_AF'
                 ]

    regional_corr = []
    regional_pval = []
    kovacs_regions = list(kovacs_regions_rating.keys())

    # for each region, calculates corr with region-specific rating
    for r in kovacs_regions:
        r_dat = dat[dat['region_name'] == r]  # get region dat
        r_stage = kovacs_regions_rating[r]  # get region-specific rating
        r_dat_ = r_dat[~r_dat[r_stage].isna()]  # get rid of NAN
        r_corr = [round(spearmanr(r_dat_[[i,
                                          r_stage]])[0], 3) for i in tau_types]
        p_val = [round(spearmanr(r_dat_[[i,
                                          r_stage]])[1], 3) for i in tau_types]
        regional_corr.append(r_corr)
        regional_pval.append(p_val)
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                kovacs_regions[0]: regional_corr[0],
                                kovacs_regions[0]+'_pval': regional_pval[0],
                                kovacs_regions[1]: regional_corr[1],
                                kovacs_regions[1]+'_pval': regional_pval[1],
                                kovacs_regions[2]: regional_corr[2],
                                kovacs_regions[2]+'_pval': regional_pval[2],
                                kovacs_regions[3]: regional_corr[3],
                                kovacs_regions[3]+'_pval': regional_pval[3],
                                kovacs_regions[4]: regional_corr[4],
                                kovacs_regions[4]+'_pval': regional_pval[4],
                                kovacs_regions[5]: regional_corr[5],
                                kovacs_regions[5]+'_pval': regional_pval[5]
                                })
    print('Correlation between region-specific tau & region-specific rating')
    return output