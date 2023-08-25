import os
import gzip
import json
import numpy as np
import random
import pandas as pd

def main():
    
    max_steps = 500
    
    ##### V2 2022
    ######### Orasem
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/analysis_1k/val/results/val/stats_all_1677614125.2470667.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/analysis/val/results/val/stats_all_1677627610.434429.csv'
    
    ######### PredSem
    ############### RedNet
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/rednet/results/val/stats_all_1677991890.9602792.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/predsem/rednet/map_res_0.5/results/val/stats_all_1691462494.552468.csv'
    
    ############### Detic
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/iccv_rebuttal/detic/val_split/res_0.2/results/val/stats_all_1685047860.8896706.csv'
    
    ##### V1 2021
    ########## OraSem
    ############### +SemExp
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v1/orasem/semexp/results/val/stats_all_1691628563.9681735.csv'
    
    ########## RedNet
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v1/predsem/rednet/map_res_0.5/results/val/stats_all_1691457647.3578885.csv'
    
    ########## Detic
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/iccv_rebuttal/chal2021/detic/val_split/grid_sd_0.5/res_0.2/results/val/stats_all_1685047213.342055.csv'
    
    ########## RedNet + ANS global policy
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v1/orasem/activeneural/results/val/stats_all_1691614585.5868964.csv'
    
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/rednet/results/val/stats_all_1677982394.5326731.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/rednet/results/val/stats_all_1678029447.286416.csv'
    
    # Frontier
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/analysis/w_dilation/val/results/val/stats_all_1678308234.0525942.csv'
    
    # RedNet
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/rednet/results/val/stats_all_1678029447.286416.csv'
    
    # Detic
    
    ##--------------- new
    #####CHAL 2021
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/iccv_rebuttal/chal2021/detic/val_split/grid_sd_0.5/res_0.2/results/val/stats_all_1685047213.342055.csv'
    
    # chal 2023
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/iccv_rebuttal/detic/val_split/res_0.2/results/val/stats_all_1685047860.8896706.csv'
    
    # stubborn
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/analysis/stubborn/val/results/val/stats_all_1678247567.7356858.csv'
    
    ########
    
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/detic/debug/results/val/stats_all_1678816358.0136852.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/detic/custom_vocab/results/val/stats_all_1678818560.527842.csv'
    #### w/ mean goal selection
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/detic/custom_vocab/w_mean_selection/results/val/stats_all_1678822948.2412186.csv'
    #### w/ tv
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/detic/custom_vocab/w_mean_selection/w_tv/results/val/stats_all_1678834022.9860442.csv'
    #### w/o mean goal selection
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/predsem/detic/custom_vocab/wo_mean_selection/w_tv/results/val/stats_all_1678844315.7692888.csv'
    
    # Stubborn
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/pointnav/hm3d-depth/objnav/analysis/stubborn/val/results/val/stats_all_1678247567.7356858.csv'
    
    ###### HM3D_0.2 v2
    #### orasem
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/objnav_v2.0/results/val/stats_all_1691776677.1554513.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/objnav_v2.0/results/val/stats_all_1691781289.1225533.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/objnav_v2.0/not_first_goal/height_0.88/results/val/stats_all_1691797364.2024558.csv'
    
    #### detic + semexp
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/onav_v2/predsem/semexp/results/val/stats_all_1691777184.8791404.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/objnav_v2.0/not_first_goal/results/val/stats_all_1691796458.2587438.csv'
    
    ###### HM3D_0.2 v3
    #### orasem
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/results/val/stats_all_1691723980.6529274.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/stop_0.9/results/val/stats_all_1691774504.217546.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/orasem/no_stop/results/val/stats_all_1691775604.4173598.csv'
    
    ### detic + semexp
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/detic/semexp/results/val/stats_all_1691724160.0760832.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/detic/rand_50/results/val/stats_all_1691771531.320771.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/detic/rand_20/results/val/stats_all_1691774590.1625445.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/detic/random/rgb2bgr/results/val/stats_all_1692240672.7640033.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/detic/random/rgb2bgr/stop_at_0.9/results/val/stats_all_1692241699.3672633.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/home_robot/pointnav/sem_exp/results/val/stats_all_1692223013.0462742.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/detic/random/rgb2bgr/stop_at_0.9/results/val/stats_all_1692250904.564762.csv'
    
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/home_robot/pointnav/random/results/val/stats_all_1692237672.1742103.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/home_robot/pointnav/random/no_stop/results/val/stats_all_1692239865.4202547.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/home_robot/pointnav/random/stop_at_0.9/results/val/stats_all_1692241585.5051696.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v2/predsem/home_robot/pointnav/sem_exp_policy/stop_at_0.9/results/val/stats_all_1692249511.7795794.csv'
    
    file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v1/predsem/detic/random/rgb/results/val/stats_all_1692833666.973552.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v1/predsem/detic/random/rgb/results/val/stats_all_1692834387.3377337.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v1/predsem/detic/random/rgb/results/val/stats_all_1692837896.5190315.csv'
    # file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v1/predsem/detic/random/rgb/model2/results/val/stats_all_1692847026.1935294.csv'
    file_name = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/experiments/wacv/v2/hm3d_v1/predsem/detic/random/rgb/results/val/stats_all_1692847836.938631.csv'
    
    df = pd.read_csv(file_name, sep=",", header=0, index_col=False) #.head(n=21)
    # df = df.head(405)
    print(f"{len(df)} total episodes.")
    # print(df.columns)
    # print(df.loc[0])
    
    df_success = df[df["success"]==1]
    print(f"{len(df_success)} episodes succeeded ({len(df_success)*100/len(df)}).")
    num_failed = len(df)-len(df_success)
    print(f"{num_failed} episodes failed ({num_failed*100/len(df)}).")
    
    m = df["spl"].mean()
    print("-------------------------------")
    print(f"Avg SPL is {m}")
    print(f"Avg SoftSPL is {df['softspl'].mean()}")
    print(f"Avg distance_to_goal is {df['distance_to_goal'].mean()}")
    print(f"Avg success is {df['success'].mean()}")
    d = df["episode_length"].mean()
    print(f"Avg episode_length is {d}")
    print("-------------------------------")
    # print(f"Avg prediction_accuracy_bg_obj is {df['prediction_accuracy_bg_obj'].mean()}")
    # print(f"Avg prediction_accuracy_goal is {df['prediction_accuracy_goal'].mean()}")
    # print(f"Avg prediction_accuracy_obj is {df['prediction_accuracy_obj'].mean()}")
    # print(f"Avg prediction_precision is {df['prediction_precision'].mean()}")
    # print(f"Avg prediction_recall is {df['prediction_recall'].mean()}")
    # print(f"Avg prediction_f1_score is {df['prediction_f1_score'].mean()}")
    # print(f"Avg goal_seen is {df['goal_seen'].mean()}")
    # print("-------------------------------")

    # ############################ 
    # # Agent ran out of time
    # df_max_step_limit = df[df["episode_length"]==max_steps]
    # num_max_step = len(df_max_step_limit)
    # print(f"---{num_max_step} ran out of maximum step limit. ({num_max_step*100/len(df)}% of total episodes)")
    
    # # Goal not seen
    # df_goal_not_seen = df[(df["goal_seen"]==0) & (df["episode_length"]==max_steps)]
    # num_goal_not_seen = len(df_goal_not_seen)
    # print(f"---In {num_goal_not_seen} episodes, the goal was not seen. ({num_goal_not_seen*100/len(df)}% of total episodes)")
    
    # # Agent ran out of time but discovered the goal
    # df_max_step_limit_but_goal_seen = df[(df["goal_seen"]==1) & (df["episode_length"]==max_steps)]
    # num_max_step_limit_but_goal_seen = len(df_max_step_limit_but_goal_seen)
    # print(f"---In {num_max_step_limit_but_goal_seen} episodes, the agent has seen the goal but not stopped. ({num_max_step_limit_but_goal_seen*100/len(df)}% of total episodes)")
    
    # # Agent stopped at more than success distance from the goal
    # df_did_not_stop_at_goal = df[(df["goal_seen"]==1) & (df["success"]==0) & (df["episode_length"]<max_steps) & (df['distance_to_goal']>1.0)]
    # num_did_not_stop_at_goal = len(df_did_not_stop_at_goal)
    # print(f"---In {num_did_not_stop_at_goal} episodes, the agent has seen the goal but stopped at more than 1m from the goal. ({num_did_not_stop_at_goal*100/len(df)}% of total episodes).") 
    # print(f"---------For these, avg. distance_to_goal={df_did_not_stop_at_goal['distance_to_goal'].mean()}")
    
    # # Stopped at a different goal
    # df_stopped_at_diff_goal = df[(df["goal_seen"]==0) & (df["episode_length"]<max_steps)]
    # num_df_stopped_at_diff_goal = len(df_stopped_at_diff_goal)
    # print(f"---In {num_df_stopped_at_diff_goal} episodes, the agent stopped at a different goal. ({num_df_stopped_at_diff_goal*100/len(df)}% of total episodes).")
    
    # print("-------------------------------")
    
    # ############################
    # df_max_step_limit = df[df["episode_length"]==max_steps]
    # num_max_step = len(df_max_step_limit)
    # print(f"---{num_max_step} episodes failed due to maximum step limit. ({num_max_step*100/num_failed}% of failed episodes)")
    
    # df_max_step_limit_goal_d_lt = df[(df["episode_length"]==max_steps) & (df["euc_distance_to_goal_obb"]<1.0)]
    # num_max_step_limit_goal_d_lt = len(df_max_step_limit_goal_d_lt)
    # print(f"-------Among these, for {num_max_step_limit_goal_d_lt} episodes goal was within 1m. ({num_max_step_limit_goal_d_lt*100/num_max_step}%)")
    # print(f"--------------Avg. euc_distance_to_goal_obb = {df_max_step_limit_goal_d_lt['euc_distance_to_goal_obb'].mean()}")
    
    # df_max_step_limit_goal_visible = df[(df["episode_length"]==max_steps) & (df["pixel_cov_of_goal"]>0)]
    # num_max_step_limit_goal_visible = len(df_max_step_limit_goal_visible)
    # print(f"-------and for {num_max_step_limit_goal_visible} episodes goal was visible. ({num_max_step_limit_goal_visible*100/num_max_step}%)")
    
    # print("-------------------------------")
    # ############################
    
    # df_success_dist_lt = df[(df["episode_length"]<max_steps) & (df["success"]==0)]
    # num_failed_suc_dist = len(df_success_dist_lt)
    # print(f"---{num_failed_suc_dist} episodes failed due to distance to goal being more than success distance from the viewpoints. ({(num_failed_suc_dist*100/num_failed)}% of failed episodes).") 
    
    # df_dist_to_obb_lt = df[(df["episode_length"]<max_steps) & (df["success"]==0) & (df["euc_distance_to_goal_obb"]<1.0)]
    # num_failed_suc_dist_obb_dist = len(df_dist_to_obb_lt)
    # print(f"-------Among them, for {num_failed_suc_dist_obb_dist} ({(num_failed_suc_dist_obb_dist*100/num_failed_suc_dist)}) episodes, the Euclidean distance to the object OBB is less than 1.0.")
    
    # df_goal_visible = df[(df["episode_length"]<max_steps) & (df["success"]==0) & (df["euc_distance_to_goal_obb"]<1.0) & (df["pixel_cov_of_goal"]>0)]
    # num_goal_visible = len(df_goal_visible)
    # print(f"--------------Among them, for {num_goal_visible} ({num_goal_visible*100/num_failed_suc_dist_obb_dist}%) episodes, goal was visible.")
    
    # mean_pixel_coverage_of_goal_at_end = df_goal_visible["pixel_cov_of_goal"].mean()
    # print(f"--------------For these episodes, average pixel visibility of the goal is {mean_pixel_coverage_of_goal_at_end} at the last step.")
    # ############################
    
    # ############################ 
    # ## Objects failed to detect
    # df['episode_name'] = df["episode_id"].str.split(pat='/').str[5]
    # df['episode_name_id'] = df['episode_name'].str.split(pat='_').str[1]
    # df['scene_name'] = df['episode_name'].str.split(pat='.').str[0]
    # df['object_category'] = ''
    
    # data_path = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/data/datasets/objectnav/hm3d/v1/val/content'
    # # data_path = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/data/datasets/objectnav/mp3d/val/content'
    # dir_list = os.listdir(data_path)
    
    # scenes = {}
    # for path in dir_list:
    #     s_name = path.split('.')[0]
    #     # print(f'++++++++path={path}')
    #     if "json.gz" in path:
    #         with gzip.open(os.path.join(data_path,path), "rt") as _f:
    #             deserialized = json.loads(_f.read())
    #             for i, episode in enumerate(deserialized["episodes"]):
    #                 # print('------------------------',str(i))
    #                 # print(f'--------s_name={s_name}; episode={str(i)}')
    #                 df.loc[(df["scene_name"]==s_name) & (df["episode_name_id"]==str(i)), "object_category"] = episode["object_category"]
    
    # # Goal detected correctly
    # df_goal_detected = df[df["success"]==1]
    # df_goal_detected = df_goal_detected[df_goal_detected['goal_seen']==1].groupby(['object_category'])
    # print(f"-----------{len(df_goal_detected)} episodes have correctly detected goal.")
    # print(df_goal_detected['object_category'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(n=20).to_string(index=False))
    
    # # Goal not detected
    # df_goal_not_detected = df[(df["success"]==0) & (df['goal_seen']==0)].groupby(['object_category'])
    # print(f"-----------{len(df_goal_not_detected)} episodes have not detected goal.")
    # print(df_goal_not_detected['object_category'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(n=20).to_string(index=False))
    
    # # Goal wrongly detected
    # df_goal_wrongly_detected = df[(df["success"]==0) & (df['goal_seen']==1)].groupby(['object_category'])
    # print(f"-----------{len(df_goal_wrongly_detected)} episodes have wrongly detected goal.")
    # print(df_goal_wrongly_detected['object_category'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(n=30).to_string(index=False))
    
    ############################ 
    
    # # Goal not detected
    # df_goal_not_detected = df[df['goal_seen']==0].groupby(['object_category'])
    # print(f"-----------{len(df[df['goal_seen']==0])} episodes have not detected goal.")
    # print(df_goal_not_detected['object_category'].count().reset_index(name='count').sort_values(['count'], ascending=False).head())
    
    # # Goal detected
    # df_goal_detected = df[df['goal_seen']==1].groupby(['object_category'])
    # print(f"-----------{len(df[df['goal_seen']==1])} episodes have  detected goal.")
    # print(df_goal_detected['object_category'].count().reset_index(name='count').sort_values(['count'], ascending=False).head())
    
    

if __name__=='__main__':
    main()