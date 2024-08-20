function interpolate_trajectory(x_start, x_end, t_vec)
    # Interpolate between two points
    # x_start: initial point
    # x_end: final point
    # t_vec: time vector
    # return: interpolated trajectory
    x_traj = zeros(length(t_vec), length(x_start))
    for i in 1:length(x_start)
        x_traj[:, i] = x_start[i] .+ (x_end[i] - x_start[i]) .* (t_vec .- t_vec[1]) ./ (t_vec[end] - t_vec[1])
    end
    return x_traj
end