from numba  import njit, prange
import numpy as np
FLOAT_TYPE = np.float32
INT_TYPE = np.int32
@njit
def compute_initial_A(epsilon,delta,
                      tm = 100,ms = 1):
    """ 
        决策阈值 A 是自适应 SPRT 的唯一参数
        将求解的平均时间表示为 A 的函数:
            t(A) = 1/(P_g*(1-1/A))*(t_m+mean_m_s*log(A)/C)
                 = (K1+K2*log(A))/(1-1/A)
        求导迭代获得A的最优值
        A* = K1/K2 + 1 + log(A*) = t_m*C/mean_m_s + 1 + log(A*)
        该迭代四次内可收敛
        感兴趣的解是 A*>1
            A0 = K1/K2 + 1
        K1 = tm/P_g
        K2 = mean_m_s/(P_g*C)
        mean_m_s 为每个样本验证的模型数量
        C = (1-delta)*log((1-delta)/(1-epsilon)) + delta*log(delta/epsilon)
        
        根据公式(10)初始化阈值A（简化迭代计算）"""
    
    C= (1 - delta) * np.log((1 - delta)/(1 -epsilon))+delta * np.log(delta/epsilon)
    K12 = tm* C / ms  # 假设tm=1, mS=1
    #K2 = ms / (epsilon**m * C)
    A = K12 + 1
    for _ in range(5):  # 迭代4次收敛
        A = K12 + 1 + np.log(A)
    return A


@njit(parallel=True)
def calc_h(eps, eps_i, del_i, l):
    h_i = np.zeros(l,dtype=FLOAT_TYPE)
    for i in prange(l):
        h_i[i] = bisection(eps, eps_i[i], del_i[i], 0.001, 15)
    return h_i
@njit
def max_iters_sprt(eps, eps_i, del_i, k_i, A_i, l,m = 1,eta_0 = 0.05):
    h_i = calc_h(eps, eps_i, del_i, l)
    P_g = eps**m 
    eta_l = 0
    for i in range(l):
        eta_l += k_i[i]*np.log(1-P_g*(1-1/(A_i[i]**h_i[i]))) #实际上到 l-1
    return np.ceil((np.log(eta_0)-eta_l)/np.log(1-P_g/A_i[l]))
@njit
def eq9(h, eps, eps_i, del_i): #eq 9
    return eps * (del_i / eps_i) ** h + (1 - eps) * ((1 - del_i) / (1 - eps_i)) ** h - 1

@njit
def bisection(eps, eps_i, del_i, a, b, tol=1e-6, max_iter=2000):
    fa = eq9(a, eps, eps_i, del_i)
    fb = eq9(b, eps, eps_i, del_i)
    if fa * fb > 0:
        return 0.0  # 无解
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = eq9(c, eps, eps_i, del_i)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c  # 达到最大迭代次数



@njit
def SPRT_RANSAC_RAW(points, 
                      threshold=0.1, 
                      eta0=0.05, 
                      epsilon = 0.05,
                      delta = 0.1 ,
                      max_iterations=10000,
                      m=3,
                      tm = 100,
                      ms = 1
                      ): 
    """ (AI Generate)Implements the SPRT (Sequential Probability Ratio Test) RANSAC algorithm 
        to fit a plane to a set of 3D points. The algorithm uses a three-point 
        sampling strategy with a maximum iteration limit.

        Parameters:
            points (numpy.ndarray): 
                A 2D array of shape (N, 3) representing the 3D points to fit the plane.
            threshold (float, optional): 
                The distance threshold to determine inliers. Default is 0.1.
            eta0 (float, optional): 
                The initial threshold for the SPRT test. Default is 0.05.
            epsilon (float, optional): 
                The initial inlier ratio estimate. Default is 0.05.
            delta (float, optional): 
                The initial probability of a point being an inlier. Default is 0.1.
            max_iterations (int, optional): 
                The maximum number of iterations to run the algorithm. Default is 10000.
            m (int, optional): 
                The number of points sampled per iteration. Default is 3.
            tm (int, optional): 
                A parameter used in the computation of the SPRT threshold. Default is 100. 
            ms (int, optional): 
                Another parameter used in the computation of the SPRT threshold. Default is 1.

        Returns:
            tuple: A tuple containing:
                - best_model (numpy.ndarray): 
                    The coefficients of the best-fit plane in the form [A, B, C, D], 
                    where the plane equation is Ax + By + Cz + D = 0.
                - best_support (int): 
                    The number of inliers supporting the best-fit plane.
                - mean_distance_error (float): 
                    The mean absolute distance of all points to the best-fit plane.

    """
    best_support = 0
    best_model = np.zeros(4,dtype=FLOAT_TYPE)
    N = points.shape[0]
    max_iterations = min(N * (N - 1) * (N - 2) // 6, max_iterations) #最大迭代次数不会超过组合数 C(n, 3)
    A_i = np.zeros(max_iterations,dtype=FLOAT_TYPE)
    k_i = np.zeros(max_iterations,dtype=INT_TYPE)
    epsilon_i = np.zeros(max_iterations,dtype=FLOAT_TYPE)
    delta_i = np.zeros(max_iterations,dtype=FLOAT_TYPE)
    A_i[0] = compute_initial_A(epsilon,delta,tm,ms)
    k_i[0] = 0
    epsilon_i[0] = epsilon
    delta_i[0] = delta
    updated_test = False #是否更新了测试
    n_rejected = 1
    test_num = 0
    #samp_list = np.zeros((max_iterations,3),INT_TYPE)
    #seen = set()
    for iter in range(max_iterations):
        #    -----------估计平面方程---------------  
        sample_indices = np.random.choice(N, 3, replace=False)
        sample_indices = np.sort(sample_indices)
        #while True:
        #    sample_indices = np.random.choice(N, 3, replace=False)
        #    sample_indices = np.sort(sample_indices)
        #    tup = (sample_indices[0], sample_indices[1], sample_indices[2])
        #    if tup not in seen:
        #        seen.add(tup)
        #        break
        p1, p2, p3 = points[sample_indices]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        D = -np.dot(normal, p1)
        model = np.array([normal[0], normal[1], normal[2], D],dtype=FLOAT_TYPE)
        k_i[test_num] += 1
        # -----------SPRT测试(Algorithm 1)---------------
        n_consitent = 0 #测试的内点数量
        lambda_ratio = 1.0 #似然比
        accepted = True
        indices = np.random.permutation(N)
        n_tested = 0 #测试的点数量
        for idx in indices:
            # ----------检查第idx个数据点是否是内点---------------
            n_tested += 1
            point = points[idx]
            distance = np.abs(normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D)/np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            is_inlier =  distance < threshold
            if is_inlier: 
                p_b,p_g = delta_i[test_num],epsilon_i[test_num]
                n_consitent += 1
            else:
                p_b,p_g = 1 - delta_i[test_num],1 - epsilon_i[test_num]
            lambda_ratio *= (p_b / p_g) #似然比计算 eq1
            if lambda_ratio > A_i[test_num]: #似然比>A : 拒绝模型 
                accepted = False
                break 
            # idx = N : 接受模型
        # -----------更新SPRT参数和平面方程参数---------------
        if not accepted: #拒绝模型 重新估计delta 如果估计delta 差别大于原始值5% 设计下一次SPRT试验
            n_rejected += 1
            delta = delta_i[test_num]*(n_rejected-1)/n_rejected + n_consitent/(n_tested*n_rejected)
            if delta > 1.05*delta_i[test_num] or delta < 0.95*delta_i[test_num]:
                updated_test = True
                test_num +=  1
                delta_i[test_num] = delta
                epsilon_i[test_num] = epsilon_i[test_num-1]
                A_i[test_num] = compute_initial_A(epsilon_i[test_num],delta_i[test_num],tm,ms)
                #k_i[test_num] = 0
            continue
        if n_consitent > best_support: #接受模型 迄今为止支持度最大 设计下一次SPRT试验
            best_support = n_consitent
            best_model = model
            new_epsilon = n_consitent / N
            if new_epsilon >=0.99: # 避免A计算报错
                break
            #if new_epsilon>epsilon_i[test_num]:
            updated_test = True
            test_num +=  1
            delta_i[test_num] = delta_i[test_num-1]
            epsilon_i[test_num] = new_epsilon
            A_i[test_num] = compute_initial_A(epsilon_i[test_num],delta_i[test_num],tm,ms)
            #k_i[test_num] = 0
        if updated_test: #更新了测试
            updated_test = False
            #if (1-epsilon_i[test_num]**m )**k_i[test_num] < eta0:  #etaR<eta0 进行才进行评估
            max_its_left = max_iters_sprt(epsilon_i[test_num],epsilon_i,delta_i,k_i, A_i, test_num,m,eta0)
            if iter> max_its_left:
                break
    dist_pt = (
            best_model[0] * points[:, 0] +
            best_model[1] * points[:, 1] +
            best_model[2] * points[:, 2] +
            best_model[3]
        )
    mean_distance_error = np.mean(np.abs(dist_pt))
    #best_inliers = np.where(dist_pt <= threshold)[0]
    return best_model, best_support,mean_distance_error

@njit
def SPRT_RANSAC_RAW_NOR(points, 
                    point_normals,
                    threshold=0.1, 
                    eta0=0.05, 
                    epsilon = 0.2,
                    delta = 0.01 ,
                    max_iterations=10000,
                    m=1,
                    tm = 100,
                    ms = 1):
    """
    (AI Generate)Implements the SPRT (Sequential Probability Ratio Test) RANSAC algorithm for plane fitting using point normals.
    This method does not perform random sampling but instead uses the normals of the points for estimation.
    The algorithm includes a maximum iteration limit.

    Parameters:
    ----------

    points : numpy.ndarray
        A 2D array of shape (N, 3) representing the 3D points to be processed.
    point_normals : numpy.ndarray
        A 2D array of shape (N, 3) representing the normals of the points.
    threshold : float, optional
        The distance threshold to determine inliers (default is 0.1).
    eta0 : float, optional
        The initial eta value for SPRT (default is 0.05).
    epsilon : float, optional
        The initial inlier ratio estimate (default is 0.2).
    delta : float, optional
        The initial probability of rejecting a good model (default is 0.01).
    max_iterations : int, optional
        The maximum number of iterations to perform (default is 10000).
    m : int, optional
        The number of points required to estimate a model (default is 1).
    tm : int, optional
        A parameter used in the computation of the SPRT threshold (default is 100).
    ms : int, optional
        A parameter used in the computation of the SPRT threshold (default is 1).

    Returns:
    -------
    best_model : numpy.ndarray
        A 1D array of shape (4,) representing the coefficients of the best-fit plane (a, b, c, d) 
        where the plane equation is ax + by + cz + d = 0.
    best_support : int
        The number of inliers supporting the best model.
    mean_distance_error : float
        The mean distance error of the points to the best-fit plane.

    """
    best_support = 0
    best_model = np.zeros(4,dtype=FLOAT_TYPE)
    N = points.shape[0]
    max_iterations = min(N, max_iterations) #最大迭代次数不会超过点的数量
    sample_indices = np.arange(N)
    A_i = np.zeros(max_iterations,dtype=FLOAT_TYPE)
    k_i = np.zeros(max_iterations,dtype=INT_TYPE)
    epsilon_i = np.zeros(max_iterations,dtype=FLOAT_TYPE)
    delta_i = np.zeros(max_iterations,dtype=FLOAT_TYPE)
    A_i[0] = compute_initial_A(epsilon,delta,tm,ms)
    k_i[0] = 0
    epsilon_i[0] = epsilon
    delta_i[0] = delta
    updated_test = False #是否更新了测试
    n_rejected = 1
    test_num = 0
    for iter in range(max_iterations):
        #    -----------估计平面方程---------------  
        p1 = points[sample_indices[iter]]
        normal = point_normals[sample_indices[iter]]
        normal /= np.linalg.norm(normal)
        D = -np.dot(normal, p1)
        model = np.array([normal[0], normal[1], normal[2], D],dtype=FLOAT_TYPE)
        k_i[test_num] += 1
        # -----------SPRT测试(Algorithm 1)---------------
        n_consitent = 0 #测试的内点数量
        lambda_ratio = 1.0 #似然比
        accepted = True
        indices = np.random.permutation(N)
        n_tested = 0 #测试的点数量
        for idx in indices:
            # ----------检查第idx个数据点是否是内点---------------
            n_tested += 1
            point = points[idx]
            distance = np.abs(normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D)/np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            is_inlier =  distance < threshold
            if is_inlier: 
                p_b,p_g = delta_i[test_num],epsilon_i[test_num]
                n_consitent += 1
            else:
                p_b,p_g = 1 - delta_i[test_num],1 - epsilon_i[test_num]
            lambda_ratio *= (p_b / p_g) #似然比计算 eq1
            if lambda_ratio > A_i[test_num]: #似然比>A : 拒绝模型 
                accepted = False
                break 
            # idx = N : 接受模型
        # -----------更新SPRT参数和平面方程参数---------------
        if not accepted: #拒绝模型 重新估计delta 如果估计delta 差别大于原始值5% 设计下一次SPRT试验
            n_rejected += 1
            delta = delta_i[test_num]*(n_rejected-1)/n_rejected + n_consitent/(n_tested*n_rejected)
            if delta > 1.05*delta_i[test_num] or delta < 0.95*delta_i[test_num]:
                updated_test = True
                test_num +=  1
                delta_i[test_num] = delta
                epsilon_i[test_num] = epsilon_i[test_num-1]
                A_i[test_num] = compute_initial_A(epsilon_i[test_num],delta_i[test_num],tm,ms)
                #k_i[test_num] = 0
            continue
        if n_consitent > best_support: #接受模型 迄今为止支持度最大 设计下一次SPRT试验
            best_support = n_consitent
            best_model = model
            new_epsilon = n_consitent / N
            if new_epsilon >=0.99: # 避免A计算报错
                break
            #if new_epsilon>epsilon_i[test_num]:
            updated_test = True
            test_num +=  1
            delta_i[test_num] = delta_i[test_num-1]
            epsilon_i[test_num] = new_epsilon
            A_i[test_num] = compute_initial_A(epsilon_i[test_num],delta_i[test_num],tm,ms)
            #k_i[test_num] = 0
        if updated_test: #更新了测试
            updated_test = False
            #if (1-epsilon_i[test_num]**m )**k_i[test_num] < eta0:  #etaR<eta0 进行才进行评估
            max_its_left = max_iters_sprt(epsilon_i[test_num],epsilon_i,delta_i,k_i, A_i, test_num,m,eta0)
            if iter> max_its_left:
                break
    dist_pt = (
            best_model[0] * points[:, 0] +
            best_model[1] * points[:, 1] +
            best_model[2] * points[:, 2] +
            best_model[3]
        )
    mean_distance_error = np.mean(np.abs(dist_pt))
    #best_inliers = np.where(dist_pt <= threshold)[0]
    return best_model, best_support,mean_distance_error

@njit
def SPRT_RANSAC_FAST(points, 
                      threshold=0.1, 
                      eta0=0.05, 
                      epsilon = 0.05,
                      delta = 0.1 ,
                      max_iterations=2000,
                      m=1,
                      tm = 100,
                      ms = 1
                      ):
    """
        (AI Generate)Implements the SPRT (Sequential Probability Ratio Test) RANSAC algorithm 
        to fit a plane to a set of 3D points. This method uses SPRT to efficiently 
        evaluate model hypotheses and avoids exhaustive sampling by leveraging 
        point normal vectors for estimation. This method is designed to be faster, but accuracy may be lower than SPRT_RANSAC_RAW.
        Parameters:
            points (numpy.ndarray): A 2D array of shape (N, 3) representing the 
                3D points to fit the plane to.
            threshold (float, optional): The distance threshold to determine 
                inliers. Defaults to 0.1.
            eta0 (float, optional): The termination threshold for the SPRT test. 
                Defaults to 0.05.
            epsilon (float, optional): The initial inlier ratio estimate. 
                Defaults to 0.05.
            delta (float, optional): The probability of a point being an inlier 
                under the null hypothesis. Defaults to 0.1.
            max_iterations (int, optional): The maximum number of iterations 
                to run the algorithm. Defaults to 2000.
            m (int, optional): The number of points required to define a model. 
                Defaults to 1.
            tm (int, optional): A parameter used in the computation of the SPRT 
                decision threshold. Defaults to 100.
            ms (int, optional): A parameter used in the computation of the SPRT 
                decision threshold. Defaults to 1.
        Returns:
            tuple: A tuple containing:
                - best_model (numpy.ndarray): The coefficients of the best-fit 
                  plane in the form [a, b, c, d], where ax + by + cz + d = 0.
                - best_support (int): The number of inliers supporting the best-fit 
                  model.
                - mean_distance_error (float): The mean absolute distance error 
                  of the points to the best-fit plane."""
    
    best_support = 0
    best_model = np.zeros(4,dtype=FLOAT_TYPE)
    #best_inliers = np.empty(0, dtype=INT_TYPE)  # 初始化内点索引数组
    N = points.shape[0]
    #id_samples_combinations = generate_combinations3(n_points,max_iterations)
    k = 0  # 迭代次数
    #max_iterations = N * (N - 1) * (N - 2) // 6  # 组合数 C(n, 3)
    max_iterations = min(N * (N - 1) * (N - 2) // 6, max_iterations) #最大迭代次数不会超过组合数 C(n, 3)
    A = compute_initial_A(epsilon,delta,tm,ms)
    n_rejected = 1
    #sample_indices = random_sort_indices(points)
    sample_indices = np.arange(N)
    s_idx = 0
    seen = set()
    for iter in range(max_iterations):
        while True:
            sample_indices = np.random.choice(N, 3, replace=False)
            sample_indices = np.sort(sample_indices)
            tup = (sample_indices[0], sample_indices[1], sample_indices[2])
            if tup not in seen:
                seen.add(tup)
                break
        p1, p2, p3 = points[sample_indices]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        D = -np.dot(normal, p1)
        model = np.array([normal[0], normal[1], normal[2], D],dtype=FLOAT_TYPE)
        # sprt_test
        support = 0 #测试的内点数量
        lambda_ratio = 1.0 #似然比
        accepted = True
        indices = np.random.permutation(N)
        n_tested = 0 #测试的点数量
        for idx in indices:
            n_tested += 1
            point = points[idx]
            distance = np.abs(normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D)/np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            is_inlier =  distance < threshold
            if is_inlier:
                p_b,p_g = delta,epsilon
                support += 1
            else:
                p_b,p_g = 1 - delta,1 - epsilon
            lambda_ratio *= (p_b / p_g)
            if lambda_ratio > A: #拒绝模型
                accepted = False
                break 
            #if is_inlier:
            #    support += 1
        if accepted and support > best_support:
            #if support > best_support:
            best_support = support
            best_model = model
            new_epsilon = support / N
            epsilon = new_epsilon
            if new_epsilon >=0.99:
                break
            A = compute_initial_A(epsilon,delta,tm,ms)
        else:
            n_rejected += 1
            delta = delta*(n_rejected-1)/n_rejected + support/(n_tested*n_rejected)
            A = compute_initial_A(epsilon,delta,tm,ms)
        #检查终止条件
        eta = (1 - (epsilon**m)*(1 - 1/A)) ** k
        if eta < eta0:
            break
        k += 1
        s_idx += 1
    dist_pt = (
            best_model[0] * points[:, 0] +
            best_model[1] * points[:, 1] +
            best_model[2] * points[:, 2] +
            best_model[3]
        )
    mean_distance_error = np.mean(np.abs(dist_pt))
    #best_inliers = np.where(dist_pt <= threshold)[0]
    return best_model, best_support,mean_distance_error

@njit
def SPRT_RANSAC_FAST_NOR(points, 
                       point_normals,
                      threshold=0.1, 
                      eta0=0.05, 
                      epsilon = 0.05,
                      delta = 0.1 ,
                      max_iterations=2000,
                      m=1,
                      tm = 100,
                      ms = 1
                      ):
    """
    (AI Generate)Implements the SPRT (Sequential Probability Ratio Test) RANSAC algorithm 
    for plane fitting using point normals for estimation. This method is designed to be faster,
    but accuracy may be lower than the original SPRT_RANSAC_RAW_NOR method.

    Parameters:
    ----------
    points : numpy.ndarray
        A 2D array of shape (N, 3) representing the 3D points to be processed.
    point_normals : numpy.ndarray
        A 2D array of shape (N, 3) representing the normal vectors of the points.
    threshold : float, optional
        The distance threshold to determine inliers. Default is 0.1.
    eta0 : float, optional
        The termination threshold for the SPRT test. Default is 0.05.
    epsilon : float, optional
        The initial inlier ratio estimate. Default is 0.05.
    delta : float, optional
        The probability of a point being an inlier under the null hypothesis. Default is 0.1.
    max_iterations : int, optional
        The maximum number of iterations to run the algorithm. Default is 2000.
    m : int, optional
        The number of points required to define a model. Default is 1.
    tm : int, optional
        A parameter used in the SPRT test. Default is 100.
    ms : int, optional
        A parameter used in the SPRT test. Default is 1.

    Returns:
    -------
    best_model : numpy.ndarray
        A 1D array of shape (4,) representing the best-fit plane model in the form [a, b, c, d],
        where ax + by + cz + d = 0.
    best_support : int
        The number of inliers supporting the best model.
    mean_distance_error : float
        The mean distance error of the points to the best-fit plane."""
    best_support = 0
    best_model = np.zeros(4,dtype=FLOAT_TYPE)
    #best_inliers = np.empty(0, dtype=INT_TYPE)  # 初始化内点索引数组
    N = points.shape[0]
    #id_samples_combinations = generate_combinations3(n_points,max_iterations)
    k = 0  # 迭代次数
    #max_iterations = N * (N - 1) * (N - 2) // 6  # 组合数 C(n, 3)
    max_iterations = min(N, max_iterations)
    A = compute_initial_A(epsilon,delta,tm,ms)
    n_rejected = 1
    #sample_indices = random_sort_indices(points)
    sample_indices = np.arange(N)
    s_idx = 0
    while True:
        p1 = points[sample_indices[s_idx]]
        normal = point_normals[sample_indices[s_idx]]
        normal /= np.linalg.norm(normal)
        D = -np.dot(normal, p1)
        model = np.array([normal[0], normal[1], normal[2], D],dtype=FLOAT_TYPE)
        # sprt_test
        support = 0 #测试的内点数量
        lambda_ratio = 1.0 #似然比
        accepted = True
        indices = np.random.permutation(N)
        n_tested = 0 #测试的点数量
        for idx in indices:
            n_tested += 1
            point = points[idx]
            distance = np.abs(normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D)/np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            is_inlier =  distance < threshold
            if is_inlier:
                p_b,p_g = delta,epsilon
                support += 1
            else:
                p_b,p_g = 1 - delta,1 - epsilon
            lambda_ratio *= (p_b / p_g)
            if lambda_ratio > A: #拒绝模型
                accepted = False
                break 
            #if is_inlier:
            #    support += 1
        if accepted and support > best_support:
            #if support > best_support:
            best_support = support
            best_model = model
            new_epsilon = support / N
            epsilon = new_epsilon
            if new_epsilon >=0.99:
                break
            A = compute_initial_A(epsilon,delta,tm,ms)
        else:
            n_rejected += 1
            delta = delta*(n_rejected-1)/n_rejected + support/(n_tested*n_rejected)
            A = compute_initial_A(epsilon,delta,tm,ms)
        #检查终止条件
        eta = (1 - (epsilon**m)*(1 - 1/A)) ** k
        if eta < eta0:
            break
        k += 1
        s_idx += 1
        if k>=max_iterations:break #强行终止
    dist_pt = (
            best_model[0] * points[:, 0] +
            best_model[1] * points[:, 1] +
            best_model[2] * points[:, 2] +
            best_model[3]
        )
    mean_distance_error = np.mean(np.abs(dist_pt))
    #best_inliers = np.where(dist_pt <= threshold)[0]
    return best_model, best_support,mean_distance_error