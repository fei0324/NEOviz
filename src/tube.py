

def createTube():
    return 

def writeTube(sampled_pts_all_list, center_all_list, time_data_list, sampled_u_all_list, sampled_v_all_list, axes_length_all, axes_direction_all, out_dir, sampled_pt_vals_all=None, img_mat_all_list=None, starting_time_index=None):
    """
    Save tube data to JSON
    starting_time_index: the starting time step index of the tube, if it is not 0.
    """
    
    data_dict = {"version": {"major": 0, "minor": 1},
                 "texture-channels": ["density", "time-delta"],
                 "polygons": []}
    num_pts_per_t = sampled_pts_all_list[0].shape[1]

    # t_min = time_lag_range[0]
    # t_max = time_lag_range[1]
    # half_range = np.max([abs(t_min), abs(t_max)])
    # print("t_min", t_min)
    # print("t_max", t_max)
    # print("half_range", half_range)

    for t, time_step in enumerate(time_data_list):
        data_dict["polygons"].append({"time": time_step})
        center_t_meters = center_all_list[t]
        center_t_meters *= astrounit.au.to(astrounit.m)
        data_dict["polygons"][t]["center"] = {"x": center_t_meters[0],
                                       "y": center_t_meters[1],
                                       "z": center_t_meters[2]}
        
        data_dict["polygons"][t]["axes-length"] = {"a": axes_length_all[t][0],
                                                   "b": axes_length_all[t][1],
                                                   "c": axes_length_all[t][2]}
        
        axes_direction_all_t = axes_direction_all[t]
        data_dict["polygons"][t]["axes-direction"] = {"x1": axes_direction_all_t[0][0],
                                                      "x2": axes_direction_all_t[0][1],
                                                      "x3": axes_direction_all_t[0][2],
                                                      "y1": axes_direction_all_t[1][0],
                                                      "y2": axes_direction_all_t[1][1],
                                                      "y3": axes_direction_all_t[1][2],
                                                      "z1": axes_direction_all_t[2][0],
                                                      "z2": axes_direction_all_t[2][1],
                                                      "z3": axes_direction_all_t[2][2]}
        
        if starting_time_index is not None:
            data_dict["polygons"][t]["texture"] = str(t + starting_time_index) + ".png"
        else:
            data_dict["polygons"][t]["texture"] = str(t) + ".png"
        
        if sampled_pt_vals_all is not None:
            sampled_pt_val_t = sampled_pt_vals_all[t]

        # img_t_dir = os.path.join(out_dir, str(t))
        # os.makedirs(img_t_dir, exist_ok=True)
        points_arr_per_t = []
        for i in range(num_pts_per_t):
            # add point coordinates
            pt = sampled_pts_all_list[t][:, i]
            pt *= astrounit.au.to(astrounit.m)
            # add texture coordinates
            texture_coord_u = sampled_u_all_list[t][i]
            texture_coord_v = sampled_v_all_list[t][i]
            points_arr_per_t.append({"x": pt[0], "y": pt[1], "z": pt[2],
                                     "u": texture_coord_v, "v": texture_coord_u, # the u and v are flipped
                                     "data": {"density": sampled_pt_val_t[i]}})
            
        # save the texture coordinate images
        if img_mat_all_list is not None:
            img_mat_t = img_mat_all_list[t]  # (res, res, 3)
            img_dir = os.path.join(out_dir, "textures")
            os.makedirs(img_dir, exist_ok=True)

            if starting_time_index is not None:
                img_name = str(t + starting_time_index) + ".png"
            else:
                img_name = str(t) + ".png"
            img_path = os.path.join(img_dir, img_name)
            # fig = plt.figure()
            plt.imshow(img_mat_t, interpolation="spline16")
            plt.axis('off')
            plt.imsave(img_path, img_mat_t)
            plt.close()
            # plt.show()
            
        # img_mat_shape = img_mat_t.shape

        # Normalize channel g based on time_lag_range so that 0 -> 0.5
        # We split the time_lag_range to positive and negative side. The larger side desides (half_range) decides the ratio
        # We use the ratio to rescale the values in g channel
        # for i in range(img_mat_shape[0]):
        #     for j in range(img_mat_shape[1]):
                # if img_mat_t[i, j, 0] != 0:
                #     print("r channel", img_mat_t[i, j, 0])
                # if img_mat_t[i, j, 1] != 0:
                #     print("g channel", img_mat_t[i, j, 1])
                # img_mat_t[i, j, 1] = img_mat_t[i, j, 1]*0.5/half_range + 0.5
                # print(img_mat_t[i, j, 1])
        # print(np.min(img_mat_t))
        # print(np.max(img_mat_t))
        # assert np.max(img_mat_t) <= 1
        # assert np.min(img_mat_t) >= 0

        data_dict["polygons"][t]["points"] = points_arr_per_t
    
    jsonpath = os.path.join(out_dir, "tube_data.json")

    with open(jsonpath, 'w') as fp:
        json.dump(data_dict, fp)
