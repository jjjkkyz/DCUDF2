# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
import math
import copy
from scipy.sparse import coo_matrix
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage import measure
import os
from dcudf2.VectorAdam import VectorAdam
import warnings
warnings.filterwarnings('ignore')


def threshold_MC(ndf, threshold, resolution,bound_min=None,bound_max=None):
    try:
        vertices, triangles,_,_ = measure.marching_cubes(
                            ndf, threshold,spacing=(2/resolution,2/resolution,2/resolution))
        vertices -= 1
        # t = vertices[:,1].copy()
        # vertices[:,1] = vertices[:,2]
        # vertices[:, 2] = -t
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
    except ValueError:
        print("threshold too high")
        mesh = None

    if bound_min is not None:
        bound_min = bound_min.cpu().numpy()
        bound_max = bound_max.cpu().numpy()
        mesh.apply_scale((bound_max-bound_min)/2)
        mesh.apply_translation((bound_min+bound_max)/2)
    mesh.apply_translation([1/resolution, 1/resolution, 1/resolution])
    mesh.apply_scale(resolution/(resolution-1))
    return mesh


def laplacian_calculation(mesh, equal_weight=True):
    """
    edit from trimesh function to return tensor
    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neightbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors
    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(np.ndarray)

    # stack neighbors to 1D arrays
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = np.ones(3)
        # the distance from verticesex to neighbors
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        # normalize group and stack into single array
        data = np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def laplacian_calculation2(mesh, equal_weight=True, selected=None):
    neighbors = mesh.vertex_neighbors # 取出顶点的邻居信息，返回的类型为，(len(self.vertices), ) int
    vertices = mesh.vertices.view(np.ndarray) # 获取三角面网格的顶点坐标，并将其转化为numpy
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])
    # 是否采用相同的权重。
    if equal_weight:
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        ones = np.ones(3)
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        data = np.concatenate([i / i.sum() for i in norms])
    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)  # 创建一个 coo格式的稀疏矩阵，即两个点之间对应的权重值，例如：matrix[0] = [x0,y0] weight0 ...
    values = matrix.data   # 从矩阵中提取非零元素值，
    indices = np.vstack((matrix.row, matrix.col))   # 创建一个新的二维数组，其中每一列都是由 稀疏矩阵matrix中非零元素的行索引和列索引构成的。

    if selected is not None:
        # 如果selected不为None，则返回[m,n]的稀疏矩阵
        selected_indices = np.array(selected)
        mask = np.isin(indices[0], selected_indices)
        selected_indices_map = {v: i for i, v in enumerate(selected_indices)}
        selected_rows = indices[0][mask]
        selected_cols = indices[1][mask]
        selected_values = values[mask]
        
        selected_rows_mapped = [selected_indices_map[v] for v in selected_rows]
        i = torch.LongTensor([selected_rows_mapped, selected_cols])
        v = torch.FloatTensor(selected_values)
        shape = (len(selected), len(vertices))
    else:
        # 如果selected为None，则返回[n,n]的稀疏矩阵
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = matrix.shape
    test = torch.sparse.FloatTensor(i, v, torch.Size(shape))    # 创建稀疏张量,其中非零元素的值和位置与之前的稀疏矩阵matrix相同
    
    return test

def laplacian_step(laplacian_op,samples):
    laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3]) - samples[:, 0:3]
    return laplacian_v

def laplacian_step2(laplacian_op,samples,selected = None):
    '''
    在对拉普拉斯的算子计算一边权值
    '''
    # torch.sparse.mm(sparse, dense): 是pytorch中的一个函数，用于计算稀疏矩阵与密集矩阵之间的乘法运算：
    #               spars: 一个稀疏矩阵，通常表示为系数张量，其中包含非零元素的索引和对应的值
    #               dense：一个密集矩阵或密集张量
    #
    # 其中： laplacian_op 是 (n,n) 的矩阵，samples 是 (n,3) 的矩阵，n个顶点， 3是xyz 
    laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3]) 
    if selected == None:
        laplacian_v = laplacian_v - samples[:, 0:3]
    else:
        laplacian_v = laplacian_v - selected[:, 0:3]

    # laplacian_v 是 拉普拉斯算子对点坐标进行加权处理后的结果，但为什么呢？
    return laplacian_v

def get_abc(vertices, faces):
    fvs = vertices[faces]
    sub_a = fvs[:, 0, :] - fvs[:, 1, :]
    sub_b = fvs[:, 1, :] - fvs[:, 2, :]
    sub_c = fvs[:, 0, :] - fvs[:, 2, :]
    sub_a = torch.linalg.norm(sub_a, dim=1)
    sub_b = torch.linalg.norm(sub_b, dim=1)
    sub_c = torch.linalg.norm(sub_c, dim=1)
    return sub_a, sub_b, sub_c


def calculate_s(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    p = (sub_a + sub_b + sub_c)/2

    s = p*(p-sub_a)*(p-sub_b)*(p-sub_c)
    s[s<1e-30]=1e-30

    sqrts = torch.sqrt(s)
    return sqrts


def get_mid(vertices, faces):
    fvs = vertices[faces]
    re = torch.mean(fvs,dim=1)
    return re


class dcudf2:
    """DCUDF mesh extraction

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            query_func:       differentiable function, input tensor[batch_size, 3] and output tensor[batch_size]
            resolution:
            threshold:
            max_iter:
            normal_step:      end of step one
            laplacian_weight:
            bound_min:
            bound_max:
            is_cut:           if model is a open model, set to True to cut double cover
            region_rate:      region of seed and sink in mini-cut
            max_batch:        higher batch_size will have quicker speed. If you GPU memory is not enough, decrease it.
            learning_rate:
            warm_up_end:
            report_freq:      report loss every {report_freq}

        """
    def __init__(self,query_func,resolution,threshold,
                 max_iter=400, normal_step=300,laplacian_weight=2000.0, bound_min=None,bound_max=None,
                 is_cut = False, region_rate=20,
                 max_batch=100000, learning_rate=0.0005, warm_up_end=25,
                 report_freq=1, watertight_separate=False,
                 sub_weight = 2, cut_weight = 3, 
                 local_weight = 0.1,local_it = 100, cut_it = 400,
                 optimization_oc = True, sub = True):
        self.u = None
        self.mesh = None
        self.device = torch.device('cuda')

        # Evaluating parameters
        self.max_iter = max_iter
        self.max_batch = max_batch
        self.report_freq = report_freq
        self.normal_step = normal_step
        self.laplacian_weight = laplacian_weight
        self.warm_up_end = warm_up_end
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.threshold = threshold
        # 100 <= cut_it < local_it < max_iter 
        self.cut_weight = cut_weight
        self.sub_weight=sub_weight
        self.local_weight = local_weight
        self.local_it = local_it
        self.cut_it = cut_it
        self.optimization_oc = True
        self.sub = sub

        if bound_min is None:
            bound_min = torch.tensor([-1+self.threshold, -1+self.threshold, -1+self.threshold], dtype=torch.float32)
        if bound_max is None:
            bound_max = torch.tensor([1-self.threshold, 1-self.threshold, 1-self.threshold], dtype=torch.float32)
        if isinstance(bound_min, list):
            bound_min = torch.tensor(bound_min, dtype=torch.float32)
        if isinstance(bound_max, list):
            bound_max = torch.tensor(bound_max, dtype=torch.float32)
        if isinstance(bound_min, np.ndarray):
            bound_min = torch.from_numpy(bound_min).float()
        if isinstance(bound_max, np.ndarray):
            bound_max = torch.from_numpy(bound_max).float()
        self.bound_min = bound_min - self.threshold
        self.bound_max = bound_max + self.threshold

        self.is_cut = is_cut
        self.region_rate = region_rate

        self.cut_time = None
        self.extract_time = None

        self.watertight_separate = watertight_separate
        if self.watertight_separate == 1:
            self.watertight_separate=True
        else:
            self.watertight_separate= False

        self.optimizer = None

        self.query_func = query_func

    def optimize(self):
        query_func = self.query_func

        u = self.extract_fields()

        self.mesh = threshold_MC(u, self.threshold, self.resolution, bound_min=self.bound_min, bound_max=self.bound_max)

        # init points
        xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
        xyz.requires_grad = True
        # set optimizer to xyz
        self.optimizer = VectorAdam([xyz])
        # init laplacian operation
        laplacian_op = laplacian_calculation(self.mesh).cuda()
        vertex_faces = np.asarray(self.mesh.vertex_faces)
        face_mask = np.ones_like(vertex_faces).astype(bool)
        face_mask[vertex_faces==-1] = False
        #yfg
        faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()
        vertices_loss = torch.zeros((len(self.mesh.vertices),1)).cuda() 
        vertices_loss.requires_grad = False
        faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()
        local_process = False   
        selected_vertex_indices = torch.arange(self.mesh.vertices.shape[0]) 
        selected_faces_indices = torch.arange(self.mesh.faces.shape[0]) 

        for it in range(self.max_iter):

            if it == 20:
                faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()

            if it % 30 == 0 and it > 0 and it <= 100:
                self.mesh.vertices = xyz.detach().cpu().numpy()
                faces_loss_numpy = faces_loss.cpu().numpy()
                faces_loss_numpy = np.squeeze(faces_loss_numpy)
                                   
                if self.sub and self.subvision(faces_loss = faces_loss_numpy, weight = self.sub_weight):
                    local_process = False
                    xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
                    xyz.requires_grad = True
                    faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()

                self.optimizer.param_groups.clear()
                self.optimizer.state.clear()
                self.optimizer.add_param_group({'params': [xyz]})
                #yfg
                vertices_loss = torch.zeros((xyz.shape[0],1)).cuda() 
                faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()
                laplacian_op = laplacian_calculation2(self.mesh).cuda()

                vertex_faces = np.asarray(self.mesh.vertex_faces)
                face_mask = np.ones_like(vertex_faces).astype(bool)
                face_mask[vertex_faces == -1] = False          

                selected_map = torch.full((self.mesh.vertices.shape[0],), -1, dtype=torch.int64)
                selected_vertex_indices = torch.arange(self.mesh.vertices.shape[0])
                selected_faces_indices = torch.arange(self.mesh.faces.shape[0]) 
            
            if it == self.local_it:
                local_process = True
                select_face_mask = torch.zeros(len(faces), dtype=torch.bool)
                mask = torch.zeros(len(faces), dtype=torch.bool)
                select_face_loss=torch.squeeze(faces_loss)
                select_face_mask[select_face_loss>(self.local_weight * select_face_loss.mean())] = True    

                select_faces_subset = faces[select_face_mask].cpu()                           
                selected_vertex_indices = torch.unique(select_faces_subset.flatten())   
                selected_faces_indices = torch.nonzero(select_face_mask).squeeze()      

                vertex_referenced = torch.zeros(len(self.mesh.vertices), dtype=bool)
                vertex_referenced[select_faces_subset] = True
                vertex_inverse = torch.zeros(len(self.mesh.vertices), dtype=torch.int64)
                vertex_inverse[vertex_referenced] = torch.arange(vertex_referenced.sum())
                selected_faces = vertex_inverse[select_faces_subset]

                mesh = self.mesh
                initial_color = [200, 200, 200, 0]  
                mesh.visual.face_colors = initial_color
                color = np.random.randint(0, 256, size=3).tolist() + [255]  
                mesh.visual.face_colors[selected_faces_indices.cpu().numpy()] = color
                laplacian_op = laplacian_calculation2(mesh = self.mesh,equal_weight = True,selected = selected_vertex_indices.cpu().numpy()).cuda()
                selected_vertex_indices_np = selected_vertex_indices.cpu().numpy()
                selected_xyz = torch.from_numpy(self.mesh.vertices[selected_vertex_indices_np].astype(np.float32)).cuda()
                selected_xyz.requires_grad = True
                self.optimizer.param_groups.clear()
                self.optimizer.state.clear()
                self.optimizer.add_param_group({'params': [selected_xyz]})
                vertices_loss = torch.zeros((selected_xyz.shape[0],1)).cuda() #yfg
                faces_loss = torch.zeros((len(selected_faces_indices),1)).cuda()
                xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
                faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()

            if it == self.cut_it:
                self.mesh.vertices = xyz.detach().cpu().numpy()
                faces_loss_numpy = faces_loss.cpu().numpy()
                faces_loss_numpy = np.squeeze(faces_loss_numpy)
                num = self.DBScan(faces_loss = faces_loss_numpy, weight = self.cut_weight)
                if num > 0:
                    xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
                    xyz.requires_grad = True
                    # set optimizer to xyz
                    faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()
                    self.optimizer.param_groups.clear()
                    self.optimizer.state.clear()
                    self.optimizer.add_param_group({'params': [xyz]})
                    #yfg
                    vertices_loss = torch.zeros((xyz.shape[0],1)).cuda() 
                    faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()
                    laplacian_op = laplacian_calculation(self.mesh).cuda()

                    vertex_faces = np.asarray(self.mesh.vertex_faces)
                    
                    face_mask = np.ones_like(vertex_faces).astype(bool)
                    face_mask[vertex_faces == -1] = False
                    selected_vertex_indices = torch.arange(self.mesh.vertices.shape[0])
                    selected_faces_indices = torch.arange(self.mesh.faces.shape[0])
            
            self.update_learning_rate(it)
            epoch_loss = 0
            self.optimizer.zero_grad()

            if local_process:
                vloss = torch.zeros((selected_xyz.shape[0],1)).cuda()
                num_samples = selected_xyz.shape[0]
            else:
                vloss = torch.zeros((xyz.shape[0],1)).cuda()
                num_samples = xyz.shape[0]
            head = 0
            while head< num_samples:
                if local_process:
                    sample_subset = selected_xyz[head: min(head + self.max_batch, num_samples)]
                else:
                    sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)
                
                vertices_tmp = df.detach().clone()
                vloss[head: min(head + self.max_batch, num_samples)] += vertices_tmp
                if vertices_loss.mean() != 0:
                    vertices_l = vertices_loss[head: min(head + self.max_batch, num_samples)]
                    ratio = vertices_l / vertices_loss.mean()
                    df = df * (ratio.clip(1, 5))
                df_loss = df.mean()
                loss = df_loss
                s_value = calculate_s(xyz, self.mesh.faces)
                face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]
                face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                face_weight = torch.sum(face_weight, dim=1)
                face_weight = torch.sqrt(face_weight.detach())
                face_weight = face_weight.max() / face_weight
                if local_process:
                    lap = laplacian_step2(laplacian_op, xyz, selected=selected_xyz)
                else:
                    lap = laplacian_step(laplacian_op, xyz)
                lap_v = torch.mul(lap, lap)
                lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
                laplacian_loss = face_weight.detach() * torch.sum(lap_v, dim=1)
                laplacian_loss_mean = laplacian_loss.mean()
                laplacian_loss = 1800 * laplacian_loss_mean
                loss = loss + laplacian_loss
                epoch_loss += loss.data 
                loss.backward()         
                head += self.max_batch
            vertices_loss += vloss
       
            if local_process:
                floss = torch.zeros((len(selected_faces_indices),1)).cuda()
                mid_num_samples = len(selected_faces_indices)
            else:
                floss = torch.zeros((len(self.mesh.faces),1)).cuda()
                mid_num_samples = len(self.mesh.faces)
            mid_head = 0                           
            while mid_head< mid_num_samples:        
                if local_process:
                    mid_points = get_mid(selected_xyz, selected_faces) 
                else:
                    mid_points = get_mid(xyz, faces) 
                sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
                mid_df = query_func(sub_mid_points)
                faces_tmp = mid_df.detach().clone() 
                floss[mid_head: min(mid_head + self.max_batch, mid_num_samples)] += faces_tmp
                if faces_loss.mean() != 0:
                    faces_l = faces_loss[mid_head: min(mid_head + self.max_batch, mid_num_samples)]
                    ratio = faces_l / faces_loss.mean()
                    mid_df = mid_df * (ratio.clip(1, 5))
                mid_df_loss = mid_df.mean()
                loss = mid_df_loss
                epoch_loss += loss.data     
                loss.backward()            
                mid_head += self.max_batch 
            faces_loss += floss

            f_loss = faces_loss.detach().repeat_interleave(3).cuda()
            v_loss = torch.zeros(len(self.mesh.vertices)).cuda()         
            v_loss.scatter_add_(0,faces[selected_faces_indices].flatten(),f_loss)    
            v_loss = (v_loss - torch.min(v_loss)) / (torch.max(v_loss) - torch.min(v_loss))
            N = copy.deepcopy(self.mesh.vertex_normals)
            N = torch.tensor(N).cuda()
            if local_process:
                N = N[selected_vertex_indices]
                v_loss = v_loss[selected_vertex_indices]
            if self.optimization_oc:
                self.optimizer.step2(N, v_loss)
            else:
                self.optimizer.step()

            if (it+1) % 10 == 0:
                if local_process:
                    xyz[selected_vertex_indices] = selected_xyz.detach()
                    points = xyz.detach().cpu().numpy()
                else:
                    points = xyz.detach().cpu().numpy()
                self.mesh.vertices = points
                self.mesh.export('experiment/optimizer/{}_Optimize.ply'.format(str(it)))


            if (it+1) % self.report_freq == 0:
                print(" {} iteration, loss={}".format(it, epoch_loss))

        final_mesh = trimesh.Trimesh(vertices=xyz.detach().cpu().numpy(), faces=self.mesh.faces, process=False)
        if self.is_cut == 1:
            from dcudf2.mesh_cut import mesh_cut
            s = time.time()
            final_mesh_cuple = mesh_cut(final_mesh,region_rate = self.region_rate)
            t = time.time()
            self.cut_time = t-s
            if final_mesh_cuple is not None:
                final_mesh_1 = final_mesh_cuple[0]
                final_mesh_2 = final_mesh_cuple[1]

                if len(final_mesh_1.vertices)>len(final_mesh_2.vertices):
                    final_mesh = final_mesh_1
                else:
                    final_mesh = final_mesh_2
            else:
                print("It seems that model is too complex, cutting failed. Or just rerunning to try again.")
        elif self.watertight_separate:
            final_mesh = self.watertight_postprocess(final_mesh)
        else:
            pass
        return final_mesh



    def extract_fields(self):

        N = 32
        X = torch.linspace(self.bound_min[0], self.bound_max[0], self.resolution).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], self.resolution).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], self.resolution).split(N)

        u = np.zeros([self.resolution, self.resolution, self.resolution], dtype=np.float32)
        # with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)

                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = self.query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        self.u = u
        return u

    def update_learning_rate(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        if iter_step>=200:
            lr *= 0.1
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def watertight_postprocess(self, mesh):
        meshes = mesh.split()
        mask = np.zeros(len(meshes))
        from pytorch3d.ops import box3d_overlap
        boxes = []
        for m in meshes:
            # if m.vertices.shape[0]<50:
            #     continue
            boxes.append(trimesh.bounds.corners(m.bounding_box.bounds))
        boxes = torch.FloatTensor(boxes)

        intersection_vol, iou_3d = box3d_overlap(boxes, boxes, eps=1e-12)
        for i in range(len(iou_3d.shape[0])):
            for j in range(i+1, len(iou_3d.shape[0])):
                if iou_3d[i][j]>0.9:
                    mask[i] = 1
        re_mesh = None
        for i in range(len(mask)):
            if mask[i] == 1:
                if re_mesh == None:
                    re_mesh=meshes[i]
                else:
                    re_mesh = re_mesh + meshes[i]
        return re_mesh

    def subvision(self, faces_loss, weight = 4):

        def calcul_normal(a,b,c):
            return np.cross(b - a, c - a)
        
        def res_normal(or_normal, trimesh_normal, a, b, c):
            dot_p1 = np.dot(or_normal, trimesh_normal)
            if dot_p1 >= 0:
                self.refine_faces.append([a, b, c])
            else:
                self.refine_faces.append([c, b, a])
        
        faces = np.asarray(self.mesh.faces) 

        vertices = np.asarray(self.mesh.vertices) 
        areas = calculate_s(torch.from_numpy(vertices),faces).numpy()  
        face_mask = np.zeros(len(faces), dtype=bool)

        face_mask[(faces_loss>(weight * faces_loss.mean())) & (areas>1.5 * areas.mean())] = True

        if not any(face_mask):
            return False, face_mask

        # the (c, 3) int array of vertex indices
        faces_subset = faces[face_mask]
        edges = np.sort(trimesh.geometry.faces_to_edges(faces_subset), axis=1)
        unique, inverse = trimesh.grouping.unique_rows(edges)   
        # then only produce one midpoint per unique edge
        mid = vertices[edges[unique]].mean(axis=1)  
        mid_idx = inverse.reshape((-1, 3)) + len(vertices)  
        face_adj2sub = self.mesh.face_adjacency[self.mesh.face_adjacency_edges_tree.query(edges)[1]].flatten() 
        refine_face_idxs = np.zeros(len(faces), dtype=int)  

        refine_face_edges = [[] for _ in range(len(faces))]           
        refine_face_unshared = [[] for _ in range(len(faces))]         
        refine_face_mid_idxs = [[] for _ in range(len(faces))]         

        face_adjacency_unshared = self.mesh.face_adjacency_unshared[self.mesh.face_adjacency_edges_tree.query(edges)[1]].flatten()

        mask_edges = np.repeat(face_mask, 3)
        all_edges = np.sort(trimesh.geometry.faces_to_edges(faces), axis=1)
        unique_edges, idx, counts = np.unique(all_edges, axis=0, return_index=True, return_counts=True)
        referenced = np.zeros(len(all_edges), dtype=bool)
        referenced[idx] = True
        inverse_idx = np.zeros(len(all_edges), dtype=np.int64)
        inverse_idx[referenced] = np.arange(referenced.sum())
        mask_edges_idx = np.ones(len(all_edges), dtype=bool)
        mask_edges_idx[idx[counts == 1]] = False
        edge_idx = np.arange(len(all_edges))[mask_edges == True]

        for i, f_idx in enumerate(face_adj2sub):
            if mask_edges_idx[edge_idx[i//2]] == True:
                refine_face_mid_idxs[f_idx].append(inverse[i//2])   
                refine_face_edges[f_idx].append(edges[i // 2])  

                refine_face_idxs[f_idx] += 1    
                refine_face_unshared[f_idx].append(face_adjacency_unshared[i])  

        refine_face_idxs[face_mask] = 0
        # the new faces_subset with correct winding
        refine_0 = np.where(refine_face_idxs == 0)[0]
        refine_1 = np.where(refine_face_idxs == 1)[0]   
        refine_2 = np.where(refine_face_idxs == 2)[0]   
        refine_3 = np.where(refine_face_idxs == 3)[0]   
        self.refine_faces = []
        self.refine_loss = []
        face_normals = self.mesh.face_normals

        for id in refine_1:
            face_mask[id] = True 
            id_loss = faces_loss[id] 
            self.refine_loss.append(id_loss) 
            self.refine_loss.append(id_loss)
            
            mid_point_1_idx = refine_face_mid_idxs[id][0] + len(vertices)
            p1,p2 = refine_face_edges[id][0]
            p3 = refine_face_unshared[id][0] 
            p1_ver = vertices[p1]
            p2_ver = vertices[p2]
            p3_ver = vertices[p3]

            or_normal = face_normals[id]
            mid_point_1_idx_ver = mid[mid_point_1_idx - len(vertices)]

            trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, p3_ver)
            trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, p3_ver)
            res_normal(or_normal, trimesh1_normal, p1, mid_point_1_idx, p3)
            res_normal(or_normal, trimesh2_normal, mid_point_1_idx, p2, p3)

        for id in refine_2:

            face_mask[id] = True
            id_loss = faces_loss[id]
            self.refine_loss.append(id_loss) 
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            mid_point_1_idx = refine_face_mid_idxs[id][0] + len(vertices)   
            mid_point_2_idx = refine_face_mid_idxs[id][1] + len(vertices)
            p1,p2 = refine_face_edges[id][0]
            p3 = refine_face_unshared[id][0]
            p4 = refine_face_unshared[id][1]
            p1_ver = vertices[p1]
            p2_ver = vertices[p2]
            p3_ver = vertices[p3]
            or_normal = face_normals[id]
            mid_point_1_idx_ver = mid[mid_point_1_idx - len(vertices)]
            mid_point_2_idx_ver = mid[mid_point_2_idx - len(vertices)]

            if p4 == p1:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, p3_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, mid_point_2_idx_ver, p3_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, mid_point_2_idx_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, p3)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, mid_point_2_idx, p3)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p2, mid_point_2_idx)
            else:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, mid_point_2_idx_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p3_ver, mid_point_2_idx_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, p3_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, mid_point_2_idx)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p3,  mid_point_2_idx)
                res_normal(or_normal, trimesh3_normal,mid_point_1_idx, p2, p3)

        for id in refine_3:
            face_mask[id] = True
            id_loss = faces_loss[id]
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            mid_point_1_idx = refine_face_mid_idxs[id][0] + len(vertices)
            mid_point_2_idx = refine_face_mid_idxs[id][1] + len(vertices)
            mid_point_3_idx = refine_face_mid_idxs[id][2] + len(vertices)
            p1,p2 = refine_face_edges[id][0]
            p3 = refine_face_unshared[id][0]
            p4 = refine_face_unshared[id][1]            
            p5 = refine_face_unshared[id][2]

            p1_ver = vertices[p1]
            p2_ver = vertices[p2]
            p3_ver = vertices[p3]

            or_normal = face_normals[id]

            mid_point_1_idx_ver = mid[mid_point_1_idx - len(vertices)]
            mid_point_2_idx_ver = mid[mid_point_2_idx - len(vertices)]
            mid_point_3_idx_ver = mid[mid_point_3_idx - len(vertices)]
            if p1 == p5:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, mid_point_2_idx_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, mid_point_3_idx_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, mid_point_3_idx_ver, mid_point_2_idx_ver)
                trimesh4_normal = calcul_normal(mid_point_2_idx_ver, mid_point_3_idx_ver, p3_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, mid_point_2_idx)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p2, mid_point_3_idx)
                res_normal(or_normal, trimesh3_normal,mid_point_1_idx,mid_point_3_idx,mid_point_2_idx)
                res_normal(or_normal, trimesh4_normal,mid_point_2_idx, mid_point_3_idx, p3)
            else:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, mid_point_3_idx_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, mid_point_2_idx_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, mid_point_2_idx_ver, mid_point_3_idx_ver)
                trimesh4_normal = calcul_normal(mid_point_3_idx_ver, mid_point_2_idx_ver, p3_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, mid_point_3_idx)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p2, mid_point_2_idx)
                res_normal(or_normal, trimesh3_normal,mid_point_1_idx,mid_point_2_idx,mid_point_3_idx)
                res_normal(or_normal, trimesh4_normal,mid_point_3_idx, mid_point_2_idx, p3)

        f = np.column_stack([faces_subset[:, 0],
                             mid_idx[:, 0],
                             mid_idx[:, 2],

                             mid_idx[:, 0],
                             faces_subset[:, 1],
                             mid_idx[:, 1],
                             
                             mid_idx[:, 2],
                             mid_idx[:, 1],
                             faces_subset[:, 2],
                             
                             mid_idx[:, 0],
                             mid_idx[:, 1],
                             mid_idx[:, 2]]).reshape((-1, 3))
        
        refine_faces = np.array(self.refine_faces)
        original_colors = self.mesh.visual.face_colors[~face_mask] 
        new_faces = np.vstack((faces[~face_mask], f)) 
        new_faces = np.vstack((new_faces, refine_faces))  
        # stack the new midpoint vertices on the end
        new_vertices = np.vstack((vertices, mid))  
        self.mesh = trimesh.Trimesh(new_vertices,new_faces, process=False) 

        new_color = np.zeros((len(self.mesh.faces), original_colors.shape[1])) 
        new_color[:original_colors.shape[0]] = original_colors 
        new_color[-(len(f)+len(refine_faces)):] = [0, 200, 0, 255] 

        visuals = trimesh.visual.color.ColorVisuals(vertex_colors=None, face_colors=new_color)
        self.mesh.visual = visuals
        
        return True

    def DBScan(self, faces_loss, weight = 5):

        def resort(ring_edges):
            ring_edges = torch.stack(ring_edges)
            indices = torch.where(ring_edges[0][0] == ring_edges[1])
            tmp = torch.tensor([ring_edges[0][1], ring_edges[0][0]])
            result = [tmp]
            if len(indices[0]) == 0:
                indices = torch.where(ring_edges[0][1] == ring_edges[1])
                result = [ring_edges[0]]

            for i in range(1, len(ring_edges)):
                if indices[0][0] == 0:
                    result.append(ring_edges[i])
                    next = ring_edges[i][1]
                else:
                    tmp = torch.tensor([ring_edges[i][1], ring_edges[i][0]])
                    result.append(tmp)
                    next = ring_edges[i][0]
                
                if i+1 >= len(ring_edges):
                    tmp = torch.tensor([result[-1][1],result[0][0]])
                    result.append(tmp)
                else:
                    indices = torch.where(next == ring_edges[i+1])
            result = torch.stack(result)
            return result

        def getCluster(diff_list,face_neighbor):
            cluster = []
            num = 0
            while diff_list:
                pos = diff_list.pop(0)
                clusterpoint = []
                clusterpoint.append(pos)
                indices = face_neighbor[pos]
                seedlist = []
                for idx in indices:
                    if idx in diff_list:
                        seedlist.append(idx)
                        diff_list.remove(idx)
                while seedlist:
                    p = seedlist.pop(0)
                    clusterpoint.append(p)
                    indices = face_neighbor[p]
                    for idx in indices:
                        if idx in diff_list:
                            seedlist.append(idx)
                            diff_list.remove(idx)
                if len(clusterpoint) > 20:
                    cluster.append(clusterpoint)   
                    num+=1
            return cluster, num

        def select_ring(edges):
            if edges.shape == 0:
                return 0, []
            ring_edges = []
            ring_num = 0
            visited_edges = set()
            
            for i in range(len(edges) - 1):
                if tuple(edges[i].tolist()) in visited_edges:
                    i +=1
                    if ring_num > 2:
                        return 3, []
                    continue
                visited_edges.add(tuple(edges[i].tolist()))
                current_ring_edges = []  
                start_vertex = edges[i][0]  
                current_vertex = edges[i][1]
                
                while True:
                    if start_vertex == current_vertex:
                        ring_edges.append(current_ring_edges) 
                        ring_num += 1
                        break
                    
                    subscript = torch.where(current_vertex == edges)
                    next_vertex = None
                    for s in range(len(subscript[0])):
                        if tuple(edges[subscript[0][s]].tolist()) not in visited_edges:
                            visited_edges.add(tuple(edges[subscript[0][s]].tolist()))
                            current_ring_edges.append(edges[subscript[0][s]])
                            if subscript[1][s] == 0:
                                next_vertex = edges[subscript[0][s]][1]
                            else:
                                next_vertex = edges[subscript[0][s]][0]
                            break
                    
                    if next_vertex is None:
                        break
                    current_vertex = next_vertex
            return ring_num, ring_edges

        def fill_hole(edges, vertices, mesh, mask):

            def dot_product(vector1, vector2):
                product = torch.dot(vector1, vector2)
                norm_vector1 = torch.norm(vector1)
                norm_vector2 = torch.norm(vector2)
                angle_radians = torch.acos(product / (norm_vector1 * norm_vector2))
                angle_degrees = angle_radians * 180 / torch.pi
                return torch.tensor([180 - angle_degrees])

            ring_edges = edges
            hole_edges = vertices[torch.squeeze(ring_edges)] 
            hole_vector = hole_edges[:, 0, :] - hole_edges[:, 1, :]
            hole_angle = []
            edges_num = len(ring_edges)
            for i in range(edges_num):
                vector1 = hole_vector[i]
                vector2 = hole_vector[(i+1) % edges_num]
                angle_degrees = dot_product(vector1, vector2)
                hole_angle.append(angle_degrees)
                
            hole_angle = torch.tensor(hole_angle)
            new_faces = []
            while len(hole_angle) - 3 != 0:
                min_index = torch.argmin(hole_angle)
                v1 = ring_edges[min_index][0].item()
                v2 = ring_edges[min_index][1].item()
                v3 = ring_edges[(min_index+1)%len(ring_edges)][1].item()
                v_coor1 = hole_edges[min_index][0]
                v_coor2 = hole_edges[min_index][1]
                v_coor3 = hole_edges[(min_index+1)%len(hole_edges)][1]
                new_faces.append([v1, v2, v3])
                new_redges = torch.tensor([v1,v3]).unsqueeze(0)                  
                new_hedges = torch.stack([v_coor1, v_coor3]).unsqueeze(0)       
                new_vector = v_coor1 - v_coor3     
                new_angle1 = dot_product(hole_vector[(min_index-1)%len(hole_vector)], new_vector)  
                new_angle2 = dot_product(new_vector, hole_vector[(min_index+2)%len(hole_vector)])  
                new_vector = new_vector.unsqueeze(0)
                
                if min_index == 0:
                    hole_angle = torch.cat((new_angle1, hole_angle[2:-1], new_angle2),dim=0)
                    ring_edges = torch.cat((ring_edges[:min_index], new_redges, ring_edges[min_index+2:]), dim=0)
                    hole_edges = torch.cat((hole_edges[:min_index], new_hedges, hole_edges[min_index+2:]), dim=0)
                    hole_vector = torch.cat((hole_vector[:min_index], new_vector, hole_vector[min_index+2:]), dim=0)
                elif min_index == len(ring_edges) - 1:
                    hole_angle = torch.cat((hole_angle[1:-2], new_angle1, new_angle2),dim=0)
                    ring_edges = torch.cat((ring_edges[1:-1], new_redges), dim=0)
                    hole_edges = torch.cat((hole_edges[1:-1], new_hedges), dim=0)
                    hole_vector = torch.cat((hole_vector[1:-1], new_vector), dim=0)
                else:
                    hole_angle = torch.cat((hole_angle[:min_index-1], new_angle1, new_angle2, hole_angle[min_index+2:]), dim=0) 
                    ring_edges = torch.cat((ring_edges[:min_index], new_redges, ring_edges[min_index+2:]), dim=0)
                    hole_edges = torch.cat((hole_edges[:min_index], new_hedges, hole_edges[min_index+2:]), dim=0)
                    hole_vector = torch.cat((hole_vector[:min_index], new_vector, hole_vector[min_index+2:]), dim=0)
                

            new_faces.append([ring_edges[0][0], ring_edges[1][0], ring_edges[2][0]])
            new_faces = torch.tensor(new_faces)
            return new_faces
        
        def hole_to_faces(hole):
            """
            Given a loop of vertex indices  representing a hole
            turn it into triangular faces.
            If unable to do so, return None

            Parameters
            -----------
            hole : (n,) int
            Ordered loop of vertex indices

            Returns
            ---------
            faces : (n, 3) int
            New faces
            vertices : (m, 3) float
            New vertices
            """
            hole = np.asanyarray(hole)
            # the case where the hole is just a single missing triangle
            if len(hole) == 3:
                return [hole], []
            # the hole is a quad, which we fill with two triangles
            if len(hole) == 4:
                face_A = hole[[0, 1, 2]]
                face_B = hole[[2, 3, 0]]
                return [face_A, face_B], []
            return [], []

        mesh = self.mesh.copy() 
        vertices = torch.tensor(mesh.vertices)
        faces = mesh.faces
        face_mid_points = np.sum(np.array(mesh.triangles), axis=1) 
        face_mask_loss = (faces_loss > np.mean(faces_loss) * weight).astype(int) 

        face_neighbor = []
        
        for i in range(len(face_mid_points)):
            face_neighbor.append([])
        for d in mesh.face_adjacency:
            face_neighbor[d[0]].append(d[1])
            face_neighbor[d[1]].append(d[0])

        diff_list = np.where(face_mask_loss == 1)[0].tolist()
        print("test0")
        face_res = np.where(np.array(face_mask_loss) == 1)[0].tolist()
        face_res = np.array(face_res)
        print("test0.5")
        res, cluster_num = getCluster(diff_list,face_neighbor)
        print("test1")
        initial_color = [200, 200, 200, 0]
        mesh.visual.face_colors = initial_color
        mask = np.full(len(faces), True, dtype=bool)
        print("test2")
        cluster = []
        new_faces = []
        num = 0
        for f in res:
            Dbscan_faces = faces[f] 
            Dbscan_edges = torch.tensor(trimesh.geometry.faces_to_edges(Dbscan_faces))    
            Dbscan_edges = torch.sort(Dbscan_edges, dim=1).values
         
            
            unique_edges, counts = torch.unique(Dbscan_edges, dim=0, return_counts=True)
            single_occurrence_edges = unique_edges[counts == 1]
            ring_num, ring_edges = select_ring(single_occurrence_edges) 
            if ring_num == 2:
                mask[f] = False
                for i in range(2):
                    ring_edges[i] = resort(ring_edges[i])
                    # new_faces.append(fill_hole(ring_edges[i], vertices, mesh, mask))
                    faces, new_vertices = hole_to_faces(ring_edges[i])
                    new_faces.append(faces)
                    if new_vertices:
                        mesh.vertices = np.vstack((mesh.vertices, new_vertices))
                cluster.append(f)
                color = np.random.randint(0, 256, size=3).tolist() + [255] 
                mesh.visual.face_colors[f] = color
                num += 1
            
        mesh.export('experiment/test_DBScan1.ply')
        print("select_num:{}, cluster_num:{}".format(num, cluster_num))
        print("spectral_clustering_over")
        if num > 0:

            # new_faces = torch.cat(new_faces, dim=0).cpu().numpy()
            # merged_faces = np.vstack((faces[mask], new_faces))
            # new_mesh = trimesh.Trimesh(mesh.vertices,merged_faces, process=False) 
            # new_mesh.remove_unreferenced_vertices()

            new_faces = torch.cat(new_faces, dim=0)
            new_faces = new_faces.cpu().numpy()
            
            # 将新生成的面片设置为红色
                    
            # initial_color = [0.444444, 1.0, 1.0, 1.0]  # 设置初始颜色，这里以白色为例
            # normalized_rgb = np.array(initial_color[:3]) * 255
            # final_color = list(normalized_rgb.astype(int)) + [int(initial_color[3] * 255)]  # 将RGB转为整数，保持Alpha值不变
            # mesh.visual.face_colors = final_color

            old_color = [0.44444, 1.0, 1.0, 1.0]
            new_color = [1.0, 0.066, 0.015, 1.0]
            old_rgb = np.array(old_color[:3]) * 255
            new_rgb = np.array(new_color[:3]) * 255
            old_color = list(old_rgb.astype(int)) + [int(old_color[3] * 255)]  # 将RGB转为整数，保持Alpha值不变
            new_color = list(new_rgb.astype(int)) + [int(new_color[3] * 255)]  # 将RGB转为整数，保持Alpha值不变

            new_colors = np.zeros((len(mesh.faces[mask])+len(new_faces), 4))  # 生成新的颜色信息

            new_colors[:(len(faces[mask]))] = old_color # 将原始颜色信息输入保存
            new_colors[-(len(new_faces)):] = new_color  # 将发生过变化的面片信息输入保存
            
            # 合并原始面片与新面片
            combined_faces = np.vstack((faces[mask], new_faces))
            
            # 创建带有颜色的 Trimesh 对象
            visuals = trimesh.visual.color.ColorVisuals(vertex_colors=None, face_colors=new_colors)
            new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=combined_faces, process=False)
            new_mesh.visual = visuals
            new_mesh.remove_unreferenced_vertices()
            
            # 保存带有颜色的模型
            new_mesh.export('experiment/test_new_mesh.obj')

            self.mesh=new_mesh
        for label in range(cluster_num):
            color = np.random.randint(0, 256, size=3).tolist() + [255]  # RGB + Alpha
            for f in res[label]:
                mesh.visual.face_colors[f] = color
        return num
    