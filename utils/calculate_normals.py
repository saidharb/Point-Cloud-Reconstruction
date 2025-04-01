train = PointCloudEmbeddingSequenceDataset("../data", "train")
val = PointCloudEmbeddingSequenceDataset("../data", "validation")
test = PointCloudEmbeddingSequenceDataset("../data", "test")

corrupt_files = []
counter = 0
import time
for dataset in [train, val, test]:
    start = time.time()
    print("START")
    for i, data in enumerate(dataset):
        print(i, end='\r')
        try:
            data = dataset[i]
            pc_path = dataset.get_pc_path(i)
            new_pc_path = pc_path.replace('pc_cad','pc_cad_norm')
            os.makedirs(os.path.dirname(new_pc_path), exist_ok=True)
            
            point_cloud = o3d.io.read_point_cloud(pc_path)
            point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
            point_cloud.orient_normals_consistent_tangent_plane(100)
            
            normals = np.asarray(point_cloud.normals)
            points = np.asarray(point_cloud.points)
            pc_n = np.hstack((points, normals))
            
            pcn = o3d.geometry.PointCloud()
            pcn.points = o3d.utility.Vector3dVector(points)
            pcn.normals = o3d.utility.Vector3dVector(normals)

            o3d.io.write_point_cloud(new_pc_path, pcn)
        
        except Exception as e:
            counter += 1
            corrupt_files.append(pc_path)
            
    duration = (time.time() - start)/60
    print(f"{round(duration, 2)} minutes END\n")
    
end_duration = (time.time() - start)/60
print(f"{round(end_duration, 2)} minutes FINNISH\n")
print(counter, corrupt_files)