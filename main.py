import numpy as np
import open3d as o3d


def main():
    pcd = o3d.read_point_cloud('./data/bun_zipper.ply')

    add_noise(pcd, 0.004)

    # filtering multiple times will reduce the noise significantly
    # but may cause the points distribute unevenly on the surface.
    guided_filter(pcd, 0.01, 0.1)
    guided_filter(pcd, 0.01, 0.1)
    # guided_filter(pcd, 0.01, 0.01)

    o3d.draw_geometries([pcd])


def guided_filter(pcd, radius, epsilon):
    kdtree = o3d.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ points[i] + b

    pcd.points = o3d.Vector3dVector(points_copy)


def add_noise(pcd, sigma):
    points = np.asarray(pcd.points)
    noise = sigma * np.random.randn(points.shape[0], points.shape[1])
    points += noise


if __name__ == '__main__':
    main()
