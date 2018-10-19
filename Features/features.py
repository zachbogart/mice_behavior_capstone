def basic_features(centroids, zones_dict):
    '''Calcultes the following features: Position, Speed, Direction, Safe, Rest
    Input: centroids output from code as a dataframe; zones_dict(coordinates of each arm)'''
    
    df = centroids
    
    # Get X and Y cooredinates of each zones and identify for each frame
    cl_xmin = np.min([zones_dict['CL'][0][0], zones_dict['CL'][3][0]])
    cl_xmax = np.mean([zones_dict['CL'][1][0], zones_dict['CL'][2][0]])

    cl_ymin = np.min([zones_dict['CL'][0][1], zones_dict['CL'][1][1]])
    cl_ymax = np.max([zones_dict['CL'][2][1], zones_dict['CL'][3][1]])

    cr_xmin = np.mean([zones_dict['CR'][0][0], zones_dict['CR'][3][0]])
    cr_xmax = np.max([zones_dict['CR'][1][0], zones_dict['CR'][2][0]])

    cr_ymin = np.min([zones_dict['CR'][0][1], zones_dict['CR'][1][1]]) #same as CL
    cr_ymax = np.max([zones_dict['CR'][2][1], zones_dict['CR'][3][1]])

    ob_xmin = np.min([zones_dict['OB'][0][0], zones_dict['OB'][3][0]])
    ob_xmax = np.max([zones_dict['OB'][1][0], zones_dict['OB'][2][0]])

    ob_ymin = np.min([zones_dict['OB'][0][1], zones_dict['OB'][1][1]])
    ob_ymax = np.mean([zones_dict['OB'][2][1], zones_dict['OB'][3][1]])

    ot_xmin = np.min([zones_dict['OT'][0][0], zones_dict['OT'][3][0]]) #same as OB
    ot_xmax = np.max([zones_dict['OT'][1][0], zones_dict['OT'][2][0]])

    ot_ymin = np.mean([zones_dict['OT'][0][1], zones_dict['OT'][1][1]])
    ot_ymax = np.max([zones_dict['OT'][2][1], zones_dict['OT'][3][1]])

    m_xmin = cl_xmax
    m_xmax = cr_xmin
    m_ymin = ob_ymax
    m_ymax = ot_ymin
    
    
    df['zone'] = np.where(((cl_xmin<= df['x']) & (df['x']<=cl_xmax) & (cl_ymin<= df['y']) & (df['y']<=cl_ymax)), 'CL',
                      np.where(((cr_xmin<= df['x']) & (df['x']<=cr_xmax) & (cr_ymin<= df['y']) & (df['y']<=cr_ymax)), 'CR', 
                               np.where(((ob_xmin<= df['x']) & (df['x']<=ob_xmax) & (ob_ymin<= df['y']) & (df['y']<=ob_ymax)), 'OB', 
                                        np.where(((ot_xmin<= df['x']) & (df['x']<=ot_xmax) & (ot_ymin<= df['y']) & (df['y']<=ot_ymax)), 'OT', 
                                                 np.where(((m_xmin<= df['x']) & (df['x']<=m_xmax) & (m_ymin<= df['y']) & (df['y']<=m_ymax)), 'M', np.nan))))) 
    
    # Finding movement based parameters
    
    df[['x-1', 'y-1']] = df[['x', 'y']].shift(1)
    df[['x-5', 'y-5']] = df[['x', 'y']].shift(5)
    df[['x-10', 'y-10']] = df[['x', 'y']].shift(10)
    
    frames = 30.0183 # number of frames per second
    
    #Finding velocities
    df['v_1']= np.sqrt(np.square(df['x']- df['x-1'])+ np.square(df['y']- df['y-1']))*frames
    df['v_5']= np.sqrt(np.square(df['x']- df['x-5'])+ np.square(df['y']- df['y-5']))*(frames/5.0)
    df['v_10']= np.sqrt(np.square(df['x']- df['x-10'])+ np.square(df['y']- df['y-10']))*(frames/10.0)
    
    #Other features
    
    safe_size = 0.05
    
    df['rest'] = np.where(df['v_1'] ==0, 1, 0)
    df['safe_rest'] = np.where((df['x'] < (cl_xmin+ safe_size*(cl_xmax-cl_xmin))) | (df['x'] > (cr_xmax - safe_size*(cl_xmax-cl_xmin))), 1, 0)
    df['safe_pos'] = np.where((df['x'] < (cl_xmin+ safe_size*(cl_xmax-cl_xmin))) | (df['x'] > (cr_xmax - safe_size*(cl_xmax-cl_xmin))), 1, 0)
    df['dir'] = np.where(np.abs(df['x-1']-df['x']) > np.abs(df['y-1']-df['y']), 
                    np.where(df['x-1']>df['x'], 'l', 'r'), np.where(df['y-1']> df['y'], 'd', 'u'))
    
    return df

def agg_features(df_basic):
    '''Input: df_basic output from basic_features function
    Output: t, p, v: dataframes with time, position & velocity aggregated features'''
    
    zones = ['CL', 'CR', 'OB', 'OT', 'M']
    t = pd.DataFrame(np.zeros((1,5)), columns= zones)
    v = pd.DataFrame(np.zeros((1,5)), columns= zones)
    p = pd.DataFrame(columns= zones)
    
    for zone in zones:
        t.loc[:, zone] = (df['zone'] == zone).sum()*100.0/df['zone'].isin(zones).sum()
        v.loc[:, zone] = df.loc[df['zone'] == zone, 'v_1'].mean()
        p.loc[:, zone] = (df.loc[df['zone'] == zone, 'x'].mean(), df.loc[df['zone'] == zone, 'y'].mean())

    t['rest'] = np.sum(df['rest']==1)*100.0/df['zone'].isin(zones).sum()
    t['safe'] = np.sum((df['rest']==1) & (df['safe']==1))*100.0/df['zone'].isin(zones).sum()
    t['peek'] = np.sum((df['rest']==1) & (df['x'] < cr_xmax+ 0.05*(cr_xmax-cr_xmin)) &
                      (df['x'] > cl_xmax - 0.05*(cl_xmax-cl_xmin)) & (df['y'] > ob_ymax- 0.05*(ob_ymax-ob_ymin)) &
                       (df['y'] < ot_ymax+ 0.05*(ot_ymax-ot_ymin))
                      )*100.0/df['zone'].isin(zones).sum()
    t['closed'] = t['CL'] + t['CR']
    t['open'] = t['OB'] + t['OT'] + t['M']
    
    v['CL-l'] = df.loc[(df['dir']=='l') & (df['zone'] == 'CL'),'v_1'].mean()
    v['CL-r'] = df.loc[(df['dir']=='r') & (df['zone'] == 'CL'),'v_1'].mean()
    v['CR-l'] = df.loc[(df['dir']=='l') & (df['zone'] == 'CR'),'v_1'].mean()
    v['CR-r'] = df.loc[(df['dir']=='r') & (df['zone'] == 'CR'),'v_1'].mean()
    v['OB-u'] = df.loc[(df['dir']=='u') & (df['zone'] == 'OB'),'v_1'].mean()
    v['OB-d'] = df.loc[(df['dir']=='d') & (df['zone'] == 'OB'),'v_1'].mean()
    v['OT-u'] = df.loc[(df['dir']=='u') & (df['zone'] == 'OT'),'v_1'].mean()
    v['OT-d'] = df.loc[(df['dir']=='d') & (df['zone'] == 'OT'),'v_1'].mean()
    
    return t, p, v

#Example

df_features = basic_features(df, zones_dict)
t, p ,v =  agg_features(df_features)