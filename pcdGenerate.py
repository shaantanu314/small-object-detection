import open3d as o3d
import numpy as np
import cv2
import os

def registration_result(source, target):
    # target.transform(transformation)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(source.points), np.asarray(target.points))))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack((np.asarray(source.colors), np.asarray(target.colors))))

    return pcd

# def mergePCD(source,target)

def generatePointCloud(imgpath,depthpath,posepath, cameraIntrinsicMatrix):
    '''
    The Root Dir must contain these following directories
    - /imgs
    - /poses
    - /depths
    '''
    scans_list = os.listdir(posepath)
    scans_list = list(map(lambda x: x.split('.')[0],scans_list))


    pcd_list = np.empty((0,3))
    color_list = np.empty((0,3))

    transforms = list()

    reg_p2p = None

    latest_registrations = []

    for i,scan in enumerate(scans_list):
        print(i)
        depth_file = depthpath+"/"+scan+'.npy'
        image_file = imgpath+'/'+scan+'.png'

        if(not os.path.exists(depth_file)):
            continue

        if(not os.path.exists(image_file)):
            continue
        

        try:
            depth_map = o3d.geometry.Image(np.ascontiguousarray(np.load(depth_file)).astype(np.float32))
            rgb_img = cv2.cvtColor(cv2.imread(image_file),cv2.COLOR_BGR2RGB)
        except:
            print(f"error caused due to {depth_file} or {image_file}")
            continue
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_img), o3d.geometry.Image(depth_map),convert_rgb_to_intensity=False)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cameraIntrinsicMatrix)

        # Transform the point cloud in local frame to global world coordinates
        pose = np.loadtxt(posepath+'/'+scan+'.txt',dtype=np.float32)
        pcd.transform(pose)

        # Take the each 25ht point of the pointcloud
        pcd = pcd.uniform_down_sample(100)

        # Set initial transformation and threshold. May depend on previous params
        threshold = 0.005
        trans_init = np.asarray([[1.0, 0, 0, 0],
                                 [0, 1.0, 0, 0],
                                 [0, 0, 1.0, 0],
                                 [0.0, 0.0, 0.0, 1.0]])

        if reg_p2p is not None and np.asarray(reg_p2p.points).shape != 0:

            latest_pcd = o3d.geometry.PointCloud()
            for pcd_curr in latest_registrations:
                latest_pcd = registration_result(latest_pcd, pcd_curr)

            reg_res = o3d.pipelines.registration.registration_icp(
                    latest_pcd, pcd, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
            transforms.append(reg_res.transformation)

            transformed_pcd = pcd.transform(reg_res.transformation)
            latest_registrations.append(transformed_pcd)
            reg_p2p = registration_result(reg_p2p, transformed_pcd)
        else:
            reg_p2p = pcd

        if(len(latest_registrations)>50):
            latest_registrations = latest_registrations[1:]

        color_list = np.vstack((color_list,np.asarray(pcd.colors)))
        pcd_list = np.vstack((pcd_list,np.asarray(pcd.points)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_list)
    pcd.colors = o3d.utility.Vector3dVector(color_list)

    return pcd, reg_p2p, transforms


def getCameraIntrinsicMatrix(intrinsicMatrixPath):
    camera_intrinsics = np.loadtxt(intrinsicMatrixPath).astype(np.float64)
    fx = camera_intrinsics[0][0]
    fy = camera_intrinsics[1][1]
    S = camera_intrinsics[0][1]
    cx = camera_intrinsics[0][2]
    cy = camera_intrinsics[1][2]

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=fx,fy=fy,cx=cx,cy=cy)
    
    return camera_intrinsic


