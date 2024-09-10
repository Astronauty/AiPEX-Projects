
import Pkg
Pkg.activate(@__DIR__)
# Pkg.upgrade_manifest()
# Pkg.update()
Pkg.resolve()
Pkg.instantiate()

import FiniteDiff
import ForwardDiff as FD
import Convex as cvx 
import ECOS
import Plots
import MuJoCo as MJ
# import Combinatorics

using LinearAlgebra
using Random
using JLD2
using Test
using StaticArrays
using Printf
using Distributions
using MathOptInterface
using Combinatorics

include(joinpath(@__DIR__, "utils","fmincon.jl"))

function wrapped_mj_step(model, data, xk, uk)
    """
    wrapped_mj_step(model, data, xk, uk)

    Computes the next state x_{k+1} given the current state x_k and control uk
    """
    # given the current model and data. set the state and control to the model and perform a forward step
    if typeof(xk) == Vector{Float64}
    
        data.qpos .= xk[1:model.nq]
        data.qvel .= xk[(model.nq + 1):end]
        data.ctrl .= uk

    else
        # if using diff types, we need to convert the dual numbers to floats
        # uk = ForwardDiff.value.(uk)
        # xk = ForwardDiff.value.(x)
        # # set control
        # data.ctrl[:] .= converted_uk

        # # set state
        # data.qpos .= converted_x[1:model.nq]
        # data.qvel .= converted_x[(model.nq + 1):end]

        xk = ForwardDiff.value(xk)
        uk = ForwardDiff.value(uk)

        data.qpos .= xk[1:model.nq]
        data.qvel .= xk[(model.nq + 1):end]
        data.ctrl .= uk
    end
    
    # take discrete dynamics step 
    step!(model, data) 

    # return updated state k + 1
    zkp1 = zeros(model.nq + model.nv + model.na) 
    zkp1 .= get_physics_state(model, data)

    # finger_coordinates = data.geom_xpos[finger_geom_indices,:]
    
    return zkp1
end

function dynamics(params::NamedTuple, x::Vector, u::Vector)
    model = params.model
    if typeof(x) == Vector{Float64}
    
        data.qpos .= x[1:model.nq]
        data.qvel .= x[(model.nq + 1):end]
        data.ctrl .= u

    else
        # if using diff types, we need to convert the dual numbers to floats
        x = ForwardDiff.value(x)
        u = ForwardDiff.value(u)

        data.qpos .= x[1:model.nq]
        data.qvel .= x[(model.nq + 1):end]
        data.ctrl .= u
    end
    forward!(model, data)
    # @show typeof(data.qvel)
    # @show size(data.qvel)
    # @show typeof(data.qacc)
    xdot = vec([data.qvel; data.qacc])
    return xdot
end



function hermite_simpson(params::NamedTuple, x1::Vector, x2::Vector, u, dt::Real)::Vector
    # TODO: input hermite simpson implicit integrator residual 
     x_mid = 0.5(x1 + x2) + (dt/8) * (dynamics(params, x1, u) - dynamics(params, x2, u))
     res = x1 + (dt/6) * (dynamics(params, x1, u) + 4*dynamics(params, x_mid, u) + dynamics(params, x2, u)) - x2
     return res
end

function robohand_cost(params::NamedTuple, Z::Vector)::Real
    # TODO: implement cost function
    idx, N, xg = params.idx, params.N, params.xg
    model = params.model
    data = params.data
    Q, R, Qf = params.Q, params.R, params.Qf

    # stage cost
    cost = 0.0
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]

        # @show xi-xg
        # cost += 0.5*(xi[94:96]-xg[94:96])'*Q*(xi[94:96]-xg[94:96])+0.5*ui'*R*ui
        cost += 0.5*(xi-xg)'*Q*(xi-xg)+0.5*ui'*R*ui
    end

    # terminal cost 
    xf = Z[idx.x[N]]
    # cost += 0.5*(xf[94:96]-xg[94:96])'*Qf*(xf[94:96]-xg[94:96])
    cost += 0.5*(xf-xg)'*Qf*(xf-xg)
    # @show typeof(cost)
    return cost
end

function robohand_temporal_pose_cost(params::NamedTuple, Z::Vector)::Real
    idx, N, xg= params.idx, params.N , params.xg
    model = params.model
    data = params.data
    Q, R, Qf = params.Q, params.R, params.Qf

    cost = 0.0
    
    # joint cost
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]

        if i < N/2
            resetkey!(model, data, 2)
        else
            resetkey!(model, data, 3)
        end
        goal_pose = vec([data.qpos; data.qvel])

        cost += 0.5*(xi-goal_pose)'*Qf*(xi-goal_pose)+0.5*ui'*R*ui
    end

    return cost
end

function robohand_end_effector_cost(params::NamedTuple, Z::Vector)::Real
    idx, N, xg= params.idx, params.N , params.xg
    xic = params.xic
    model = params.model
    data = params.data
    t_vec = params.t_vec
    Q, R, Qf = params.Q, params.R, params.Qf
    finger_constraints, target_site_names, fingertip_site_names = params.finger_constraints, params.target_site_names, params.fingertip_site_names

    cost = 0.0
    
    for i in 1:(N-1)
        # find the active finger constraint and target site position
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]

        data.qpos .= xi[1:model.nq]
        data.qvel .= xi[(model.nq + 1):end]
        forward!(model, data)

        finger_active = false

        for j = 1:length(finger_constraints)
            if finger_constraints[j].t_start <= t_vec[i] && t_vec[i] < finger_constraints[j].t_end # check if the finger is active at this time step
                finger_active = true

                # find position of the desired target site
                target_site_name = target_site_names[finger_constraints[j].site_index] # the position to strike with a finger
                # @show target_site_name
                target_site_pos = MJ.geom(data, target_site_name).xpos
                
                # find position of the desired finger
                fingertip_site_name = fingertip_site_names[finger_constraints[j].finger_index]
                # @show fingertip_site_name
                fingertip_site_pos = MJ.site(data, fingertip_site_name).xpos

                # println(j)
                
                cost += 0.5*(fingertip_site_pos-target_site_pos)'*Q*(fingertip_site_pos-target_site_pos) + 0.5*ui'*R*ui
            end
        end

        # if !finger_active
        #     cost += 0.5*(xi - xic)'*Qf*(xi - xic) + 0.5*ui'*R*ui
        # end
    end

    return cost

end

function robohand_dynamics_constraints(params::NamedTuple, Z::Vector)::Vector
    idx, N, dt = params.idx, params.N, params.dt
    model = params.model
    data = params.data
    # create c in a ForwardDiff friendly way (check HW0)
    c = zeros(eltype(Z), idx.nc)

    
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]] 
        xip1 = Z[idx.x[i+1]]
        
        # data.qpos[94:96] = xi[1:3]
        # data.qvel[94:96] = xi[4:6]

        ## MuJoCo Integration
        xip1_mujoco = wrapped_mj_step(model, data, xi, ui)
        c[idx.c[i]] = xip1_mujoco - xip1

        ## Hermite Simpson Integration
        # c[idx.c[i]] = hermite_simpson(params, xi, xip1, ui, dt)
    end
    # println(typeof(c))
    return c 
end


function robohand_equality_constraints(params::NamedTuple, Z::Vector)::Vector
    # TODO: implement equality constraints
    # return zeros(eltype(Z), 0)
    N, idx, xic, xg = params.N, params.idx, params.xic, params.xg 
    model = params.model
    data = params.data

    con_1 = Z[idx.x[1]] - xic
    con_2 = Z[idx.x[N]] - xg

    # return zeros(eltype(Z), 0)
    # return [con_1; con_2]
    # return [con_1; con_2; robohand_dynamics_constraints(params, Z)]
    return [con_1; robohand_dynamics_constraints(params, Z)]


end

function robohand_inequality_constraints(params::NamedTuple, Z::Vector)::Vector
    idx, N, dt = params.idx, params.N, params.dt
    model = params.model
    data = params.data
    
    N = params.N
    finger_constraints, target_site_names, fingertip_site_names = params.finger_constraints, params.target_site_names, params.fingertip_site_names

    c = zeros(eltype(Z), 5*N) # 5 finger constraints per timestep 

    # Cancel out the constraint if the finger is supposed to be playing a note at this timestep

    for i in 1:N
        # find the active finger constraint and target site position
        xi = Z[idx.x[i]]
        # ui = Z[idx.u[i]]

        data.qpos .= xi[1:model.nq]
        data.qvel .= xi[(model.nq + 1):end]
        forward!(model, data)


        fingers_vertical_pos = [] 
        for fingertip_site_name in fingertip_site_names
            push!(fingers_vertical_pos, MJ.site(data, fingertip_site_name).xpos[3])
        end


        # for j = 1:length(finger_constraints)
        #     if finger_constraints[j].t_start <= t_vec[i] && t_vec[i] < finger_constraints[j].t_end # check if the finger is active at this time step
        #        fingers_vertical_pos[finger_constraints[j].finger_index] = 10 # Cancel the constraint if the finger is active
        #     end
        # end
        c[i:i+4] = fingers_vertical_pos
    end

    return c # Scale relative to equality constraints
    # return inequality_constraints
end

function create_idx(nx,nu,N)
    # This function creates some useful indexing tools for Z 
    # x_i = Z[idx.x[i]]
    # u_i = Z[idx.u[i]]
    
    # our Z vector is [x0, u0, x1, u1, …, xN]
    nz = (N-1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1 : nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx + 1):(nx + nu)) for i = 1:(N - 1)]
    
    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx) .+ (1 : nx) for i = 1:(N - 1)]
    nc = (N - 1) * nx # (N-1)*nx 
    
    return (nx=nx,nu=nu,N=N,nz=nz,nc=nc,x= x,u = u,c = c)
end

struct FingerConstraint
    t_start::Float64
    t_end::Float64
    site_index::Int
    finger_index::Int
end

function solve_hand_dircol(θ;verbose=true)
    # instantiate model and data
    model = load_model("models/scene_right_piano_hand.xml")

    data = init_data(model)

    # reset the model and data
    reset!(model, data)

    nx = 2*model.nv
    nu = model.nu

    # initiate time and time steps
    model.opt.timestep = 0.5
    dt = model.opt.timestep
    tf = 2.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    # LQR cost
    q_diag = ones(model.nq + model.nv)
    q_diag[1:3] .= 10 # prioritize the position of the hand 
    # Q = diagm(ones(model.nq + model.nv))
    Q = diagm(q_diag)
    Qf = 0.01*Q

    # LQR cost for end-effector
    Q_cartesian = 1000*diagm(ones(3))

    R = 0.1*diagm(ones(model.nu))

    Q_neutral = Q

    # define mode constraints
    # fingertip_site_names = ["rh_shadow_hand/mfdistal_touch_site", "rh_shadow_hand/ffdistal_touch_site", "rh_shadow_hand/mfdistal_touch_site", "rh_shadow_hand/ffdistal_touch_site"] 
    fingertip_site_names = ["fingertip_thumb", "fingertip_index", "fingertip_middle", "fingertip_ring", "fingertip_little"]
    target_site_names = ["keymarker_1", "keymarker_2", "keymarker_3", "keymarker_4", "keymarker_5", "keymarker_6", "keymarker_7", "keymarker_8"]

    
    ### Mode constraints
    # finger_constraints = [
    #     FingerConstraint(0.1*tf, 0.5*tf, 1, 2),
    #     FingerConstraint(0.5*tf, 0.7*tf, 2, 3),
    #     FingerConstraint(0.7*tf, tf, 3, 4)
    # ]

    # finger_constraints = [FingerConstraint(0.1*tf, tf, 1, 1)]

    ### Octave jump
    finger_constraints = [
        FingerConstraint(0.2*tf, 0.5*tf, 1, 1),
        FingerConstraint(0.7*tf, 0.8*tf, 2, 4)
    ]


    # 5 finger scale
    # finger_constraints = []
    # start_delay = 0.1*tf
    # t_increment = (tf-start_delay)/5
    # for i = 1:5
    #     push!(finger_constraints, FingerConstraint(start_delay+t_increment*(i-1), start_delay+t_increment*i, i, (i-1)%5+1))
    # end 

    @show finger_constraints
    # sample possible fingerings for 3 notes
    finger_set = 1:5
    mode_sequences = multiset_permutations(finger_set, 3)
    
    mode_sequences_array = collect(mode_sequences)
    random_mode_sequence = rand(mode_sequences_array)

    # mode_sequence = [1, 2, 3]


    # initial and goal states
    resetkey!(model, data, 1)
    xic = vec(vcat(copy(data.qpos), copy(data.qvel)))
 
    println("Initial state: ", xic)

    # xg = vec(vcat(copy(data.qpos), copy(data.qvel)))
    xg = vec(vcat(copy(data.qpos), copy(data.qvel)))
    xg[2] = 0.1
    xg[5] = deg2rad(5)
    xg[21] = deg2rad(15)




    ### Create Indexing 
    idx = create_idx(model.nq + model.nv, model.nu, N)
    idx = create_idx(nx, nu, N)
    params = (Q = Q_cartesian, R = R, Qf = Qf, xic = xic, xg = xg, t_vec=t_vec, finger_constraints=finger_constraints, target_site_names=target_site_names, fingertip_site_names= fingertip_site_names, dt = dt, N = N, idx = idx, model=model, data=data)


    ### Primal bounds
    # x_l = vcat(model.jnt_range[:,1], -Inf*ones(model.nv), model.actuator_ctrlrange[:,1]) # combine joint pos, vel, and control limits
    # x_u = vcat(model.jnt_range[:,2], Inf*ones(model.nv), model.actuator_ctrlrange[:,2])
    x_l = -100*ones(idx.nz)
    x_u = 100*ones(idx.nz)

    # x_l = zeros(idx.nz)
    # x_u = zeros(idx.nz)
    for i = 1:N
        # x_l[idx.x[i]] = vcat(model.jnt_range[:,1], -Inf*ones(model.nv))
        # x_u[idx.x[i]] = vcat(model.jnt_range[:,2], Inf*ones(model.nv))
        x_l[idx.x[i]] = vcat(model.jnt_range[:,1], -deg2rad(10)*ones(model.nv))
        x_u[idx.x[i]] = vcat(model.jnt_range[:,2], deg2rad(10)*ones(model.nv))

        # if i < N
        #     x_l[idx.u[i]] = model.actuator_ctrlrange[:,1]
        #     x_u[idx.u[i]] = model.actuator_ctrlrange[:,2]
        # end
    end
    x_u[1] = 0.08 # prevent the wrist from going under the keybed


    ## Inequality constraints
    # c_l = 0.12 * ones(5*idx.N)
    # c_l = 0.16 * ones(5*idx.N)
    c_l= -Inf * ones(5*idx.N)
    c_u = Inf * ones(5*idx.N)
    # c_l = zeros(0)
    # c_u = zeros(0)

    ### Initial guess
    z0 = 0.0*randn(idx.nz)
    for i = 1:N
        z0[idx.x[i]] = xic
    end
    z0 += 0.01*randn(idx.nz)
    # z0 = zeros(idx.nz)


    # diff type
    # diff_type = :auto 
    diff_type = :finite
    
    # @show xic'*Q*xic

    
    Z, objs = fmincon(robohand_end_effector_cost, robohand_equality_constraints, robohand_inequality_constraints,
                x_l,x_u,c_l,c_u,z0,params, diff_type;
                tol = 1e-3, c_tol = 1e-1, dual_inf_tol = 1e1, compl_inf_tol = 1e-3, max_wall_time=120*60.0, max_iters = 1000, verbose = verbose)

    # Z = res[1]
    # objs = res[2]
    

    # pull the X and U solutions out of Z 
    X = [Z[idx.x[i]] for i = 1:N]
    U = [Z[idx.u[i]] for i = 1:(N-1)]
    return X, U, t_vec, params, objs
    # return
end