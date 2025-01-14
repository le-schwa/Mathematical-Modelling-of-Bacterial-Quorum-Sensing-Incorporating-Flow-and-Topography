module Cellular_Automata

using Plots
using Distributions
using ColorSchemes
using Random
using Colors


# Setup of the Cellular Automata

Random.seed!(123)
d = Bernoulli(0.01)

# Probability distribution of the crevices
function prob_distribution(rows,cols)
    # Initialize the probability distribution array
    prob_distribution = zeros(rows, cols)
    
    # Populate the probability distribution array
    for i in 1:rows
        for j in 1:cols
            prob_distribution[i, j] = 0.005 - (0.005 * (i - 1) / (rows-1))
        end
    end
    
    # Generate the random matrix
    return prob_distribution
end

function initialize_grid(rows, cols)
    flow_channel =  hcat(vcat((-1)*ones(cols+1)',hcat((-1)*ones(rows),Int.(rand(d, rows, cols)))),(-1)*ones(rows+1))  # Initialize grid with random binary values
    crevice_1 = vcat(Int.(rand.(Bernoulli.(prob_distribution(3*rows,Int.(cols/10))))),(-1)*ones(1,Int.(cols/10)))
    crevice_2 = vcat(Int.(rand.(Bernoulli.(prob_distribution(3*rows,Int.(cols/10))))),(-1)*ones(1,Int.(cols/10)))
    crevice = hcat((-1)*ones(3*rows+1,Int.(cols/5)),crevice_1,(-1)*ones(3*rows+1,Int.(2*cols/5)),crevice_2,(-1)*ones(3*rows+1,Int.(cols+2-cols/5-cols/10-cols/10-2*cols/5)))
    return vcat(flow_channel,crevice)
end

function initialize_nutrients(rows, cols)
    flow_channel =  hcat(vcat((-1)*ones(cols+1)',hcat((-1)*ones(rows),100*ones(rows, cols))),(-1)*ones(rows+1))  # Initialize grid with random binary values
    crevice_1 = vcat(100*ones(3*rows,Int.(cols/10)),(-1)*ones(1,Int.(cols/10)))
    crevice_2 = vcat(100*ones(3*rows,Int.(cols/10)),(-1)*ones(1,Int.(cols/10)))
    crevice = hcat((-1)*ones(3*rows+1,Int.(cols/5)),crevice_1,(-1)*ones(3*rows+1,Int.(2*cols/5)),crevice_2,(-1)*ones(3*rows+1,Int.(cols+2-cols/5-cols/10-cols/10-2*cols/5)))
    return vcat(flow_channel,crevice)
end

function initialize_nutrients_flow(rows, cols)
    flow_channel =  hcat(vcat((-1)*ones(cols+1)',hcat((-1)*ones(rows),hcat(100*ones(rows, 1)),zeros(rows,cols-1)),),(-1)*ones(rows+1))  # Initialize grid with random binary values
    crevice_1 = vcat(1000*ones(3*rows,Int.(cols/10)),(-1)*ones(1,Int.(cols/10)))
    crevice_2 = vcat(1000*ones(3*rows,Int.(cols/10)),(-1)*ones(1,Int.(cols/10)))
    crevice = hcat((-1)*ones(3*rows+1,Int.(cols/5)),crevice_1,(-1)*ones(3*rows+1,Int.(2*cols/5)),crevice_2,(-1)*ones(3*rows+1,Int.(cols+2-cols/5-cols/10-cols/10-2*cols/5)))
    return vcat(flow_channel,crevice)
end 

function initialize_AIP(rows, cols)
    flow_channel =  hcat(vcat((-1)*ones(cols+1)',hcat((-1)*ones(rows),zeros(rows,cols))),(-1)*ones(rows+1))  # Initialize grid with random binary values
    crevice_1 = vcat(zeros(3*rows,Int.(cols/10)),(-1)*ones(1,Int.(cols/10)))
    crevice_2 = vcat(zeros(3*rows,Int.(cols/10)),(-1)*ones(1,Int.(cols/10)))
    crevice = hcat((-1)*ones(3*rows+1,Int.(cols/5)),crevice_1,(-1)*ones(3*rows+1,Int.(2*cols/5)),crevice_2,(-1)*ones(3*rows+1,Int.(cols+2-cols/5-cols/10-cols/10-2*cols/5)))
    return vcat(flow_channel,crevice)
end

function plot_grid(grid)
    h = heatmap(grid[1], color = cgrad([:seashell4,:black,:red,:yellow],4,categorical = true), legend=false,yflip = true,categorical = true,clim = (-1,2))
    display(h)
end

function plot_AIP(grid)
    h = heatmap(grid[6],color = :batlowK,legend=true,yflip = true,clim=(-1,20))
    display(h)
end

function nearest_zero_value_index(matrix, start_row, start_col)
    rows, cols = size(matrix)
    visited = falses(rows, cols)
    
    # Define von Neumann neighborhood offsets
    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # BFS queue
    queue = [(start_row, start_col)]
    visited[start_row, start_col] = true
    steps = 0

    while !isempty(queue)
        current_row, current_col = popfirst!(queue)
        steps += 1

        # Check if the current element is zero
        if matrix[current_row, current_col] == 0
            return current_row, current_col
        end

        # Explore neighbors
        for (dx, dy) in offsets
            new_row = current_row + dx
            new_col = current_col + dy

            # Check if the new indices are within the matrix boundaries
            if 1 <= new_row <= rows && 1 <= new_col <= cols && !visited[new_row, new_col]
                push!(queue, (new_row, new_col))
                visited[new_row, new_col] = true
            end
        end

        if steps >= 10
            return nothing
        end
    end

    return nothing  # Return nothing if no zero value is found
end

function AIP_jump(flow::Bool)
    offsets = [(1, 0),(-1, 0),(0, 1),(0, -1)]
    if flow == true
        dx = offsets[rand(Categorical(0.5,0.5))]
        dy = rand(Poisson(100))
        return (dx[1],dy)
    else
        c = rand(Categorical(0.25,0.25,0.25,0.25)) # Diffusion
        return offsets[c]
    end
end

# Example usage
rows = 50
cols = 1000

# Example Initialization
# 1. Grid 
# 2. Bacterial Nutrients
# 3. Solution Nutrients
# 4. Flow Nutrients
# 5. Bacterial AIP 
# 6. Environment AIP
grid = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
plot_grid(grid)

"""
Iteration of the rules of the CA
"""
function bacteria_growth_iteration(grid,p_split,it,p_absorption,p_abs_c)
    new_bacteria = zeros(size(grid[1]))
    moved_AIP = zeros(size(grid[6]))
    shifted_flow = initialize_AIP(rows,cols)

    for i in collect(2:size(grid[1])[2]-1)
        for j in collect(2:size(grid[1])[1])
            if grid[1][j,i] == -1
                break
            end

            if grid[1][j,i] == 1
                grid[5][j,i] = grid[5][j,i]+1   # Production
                (dx,dy) = AIP_jump(false)
                if grid[1][j+dx,i+dy] != -1
                    moved_AIP[j+dx,i+dy] = moved_AIP[j+dx,i+dy]+1   # Movement
                end
                grid[5][j,i] = max(grid[5][j,i]-2,0)
            elseif grid[1][j,i] == 2    # Induced Bacterium
                grid[5][j,i] = grid[5][j,i]+4   # Increased Production
                (dx,dy) = AIP_jump(false)
                if grid[1][j+dx,i+dy] != -1
                    moved_AIP[j+dx,i+dy] = moved_AIP[j+dx,i+dy]+4   # Movement
                end
                grid[5][j,i] = max(grid[5][j,i]-5,0)
            end

            if grid[6][j,i] > 0
                (dx,dy) = AIP_jump((j<=rows+1))
                dx = 0
                if i+dy >= cols+1
                    grid[6][j,i] = 0
                    #=
                elseif rows+1 < j <= 2*(rows+1)
                    if grid[1][j+dx,i+dy] == -1
                        prob = 0.75
                        a = grid[6][j,i]
                        for w in 1:a
                            if rand(Bernoulli(prob)) == true
                                grid[5][j,i] = grid[5][j,i] + 1
                                grid[6][j,i] = grid[6][j,i] - 1
                            end
                        end
                        moved_AIP[rows+1,i+dy] = moved_AIP[rows+1,i+dy]+grid[6][j,i]
                        grid[6][j,i] = 0
                    end
                    =#
                elseif j+dx <= 202 && grid[1][j+dx,i+dy] != -1
                    if j<=rows+1
                        prob = p_absorption
                        #=
                    elseif rows+1 < j <= 2*(rows+1)
                        prob = 0.01+(j-rows-1)*(0.04/((2*rows+1)-(rows-1)))
                        =#
                    else
                        prob = p_abs_c
                    end
                    a = grid[6][j,i]
                    for w in 1:a
                        if rand(Bernoulli(prob)) == true
                            grid[5][j,i] = grid[5][j,i] + 1
                            grid[6][j,i] = grid[6][j,i] - 1
                        end
                    end
                    moved_AIP[j+dx,i+dy] = moved_AIP[j+dx,i+dy]+grid[6][j,i]
                    grid[6][j,i] = 0
                end
            end

            if grid[5][j,i] >= 25 && grid[1][j,i] == 1
                grid[1][j,i] = 2
            elseif grid[5][j,i] < 25 && grid[1][j,i] == 2
                grid[1][j,i] = 1
            end

            if grid[4][j,i] >= 0 && j<=rows+1
                shifted_flow[j,i+1] = grid[4][j,i]
            end

            if grid[1][j,i] > 0 && grid[4][j,i] > 0
                grid[2][j,i] = grid[2][j,i]+1
                grid[4][j,i] = grid[4][j,i]-1
            elseif grid[1][j,i] > 0 && grid[3][j,i] > 0
                grid[2][j,i] = grid[2][j,i]+1
                grid[3][j,i] = grid[3][j,i]-1
            end

            
            if mod(it,2) == 0   # timescale
                if (grid[1][j,i] == 1 || grid[1][j,i] == 2) && new_bacteria[j,i] == 0 && grid[2][j,i] > 10
                    if rand(Bernoulli(p_split)) == true
                        new_place = nearest_zero_value_index(grid[1], j, i)
                        if !isnothing(new_place)
                            new_bacteria[new_place[1],new_place[2]] = 1
                            grid[1][new_place[1],new_place[2]] = 1
                            grid[2][j,i] = grid[2][j,i]-10
                            grid[2][new_place[1],new_place[2]] = 0
                        end
                    end
                end
            end
            
        end
    end

    shifted_flow[:,1] .= -1
    shifted_flow[:,2] .= 1000
    shifted_flow[:, end] = grid[1][:, end]

    return [grid[1],grid[2],grid[3],shifted_flow,grid[5],grid[6]+moved_AIP] 
end

"""
Iteration of the adapted rules of the CA
"""
function bacteria_growth_iteration_adap(grid,p_split,it,p_absorption,p_abs_c,p_abs_c1)
    new_bacteria = zeros(size(grid[1]))
    moved_AIP = zeros(size(grid[6]))
    shifted_flow = initialize_AIP(rows,cols)

    for i in collect(2:size(grid[1])[2]-1)
        for j in collect(2:size(grid[1])[1])
            if grid[1][j,i] == -1
                break
            end

            if grid[1][j,i] == 1
                grid[5][j,i] = grid[5][j,i]+1   # Production
                (dx,dy) = AIP_jump(false)
                if grid[1][j+dx,i+dy] != -1
                    moved_AIP[j+dx,i+dy] = moved_AIP[j+dx,i+dy]+1   # Movement
                end
                grid[5][j,i] = max(grid[5][j,i]-2,0)
            elseif grid[1][j,i] == 2    # Induced Bacterium
                grid[5][j,i] = grid[5][j,i]+4   # Increased Production
                (dx,dy) = AIP_jump(false)
                if grid[1][j+dx,i+dy] != -1
                    moved_AIP[j+dx,i+dy] = moved_AIP[j+dx,i+dy]+4   # Movement
                end
                grid[5][j,i] = max(grid[5][j,i]-5,0)
            end

            if grid[6][j,i] > 0
                (dx,dy) = AIP_jump((j<=2*(rows+1)))
                dx = 0
                if i+dy >= cols+1
                    grid[6][j,i] = 0
                elseif rows+1 < j <= 2*(rows+1)
                    if grid[1][j+dx,i+dy] == -1
                        prob = p_abs_c1
                        a = grid[6][j,i]
                        for w in 1:a
                            if rand(Bernoulli(prob)) == true
                                grid[5][j,i] = grid[5][j,i] + 1
                                grid[6][j,i] = grid[6][j,i] - 1
                            end
                        end
                        moved_AIP[rows+1,i+dy] = moved_AIP[rows+1,i+dy]+grid[6][j,i]
                        grid[6][j,i] = 0
                    end
                elseif (j > 2*(rows+1) || j<=rows+1)  && j+dx <= 202 && grid[1][j+dx,i+dy] != -1
                    if j<=rows+1
                        prob = p_absorption
                        #=
                    elseif rows+1 < j <= 2*(rows+1)
                        prob = 0.01+(j-rows-1)*(0.04/((2*rows+1)-(rows-1)))
                        =#
                    else
                        prob = p_abs_c
                    end
                    a = grid[6][j,i]
                    for w in 1:a
                        if rand(Bernoulli(prob)) == true
                            grid[5][j,i] = grid[5][j,i] + 1
                            grid[6][j,i] = grid[6][j,i] - 1
                        end
                    end
                    moved_AIP[j+dx,i+dy] = moved_AIP[j+dx,i+dy]+grid[6][j,i]
                    grid[6][j,i] = 0
                end
            end

            if grid[5][j,i] >= 25 && grid[1][j,i] == 1
                grid[1][j,i] = 2
            elseif grid[5][j,i] < 25 && grid[1][j,i] == 2
                grid[1][j,i] = 1
            end

            if grid[4][j,i] >= 0 && j<=rows+1
                shifted_flow[j,i+1] = grid[4][j,i]
            end

            if grid[1][j,i] > 0 && grid[4][j,i] > 0
                grid[2][j,i] = grid[2][j,i]+1
                grid[4][j,i] = grid[4][j,i]-1
            elseif grid[1][j,i] > 0 && grid[3][j,i] > 0
                grid[2][j,i] = grid[2][j,i]+1
                grid[3][j,i] = grid[3][j,i]-1
            end

            
            if mod(it,2) == 0   # timescale
                if (grid[1][j,i] == 1 || grid[1][j,i] == 2) && new_bacteria[j,i] == 0 && grid[2][j,i] > 10
                    if rand(Bernoulli(p_split)) == true
                        new_place = nearest_zero_value_index(grid[1], j, i)
                        if !isnothing(new_place)
                            new_bacteria[new_place[1],new_place[2]] = 1
                            grid[1][new_place[1],new_place[2]] = 1
                            grid[2][j,i] = grid[2][j,i]-10
                            grid[2][new_place[1],new_place[2]] = 0
                        end
                    end
                end
            end
            
        end
    end

    shifted_flow[:,1] .= -1
    shifted_flow[:,2] .= 1000
    shifted_flow[:, end] = grid[1][:, end]

    return [grid[1],grid[2],grid[3],shifted_flow,grid[5],grid[6]+moved_AIP] 
end

"""
Applying the rules up to time point T 
"""
function cellular_automata(T,p_absorption,p_abs_c)
    grid = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    for i in 1:T
        grid = bacteria_growth_iteration(grid,0.1,i,p_absorption,p_abs_c)
        
        # Adaptation of the CA
        # grid = bacteria_growth_iteration_adap(grid,0.1,i,p_absorption,p_abs_c,0.1)
        plot_grid(grid)
    end
    plot_grid(grid)
    return grid
end

# Example
@time g = cellular_automata(400,0.15,0.1)
plot_AIP(g)
plot_grid(g)

function move_values_right(matrix::Matrix{T}) where T
    nrows, ncols = size(matrix)
    shifted_matrix = similar(matrix, T)

    for i in 1:nrows
        shifted_matrix[i, 2] = 100  # Set the first element to 100
        for j in 2:ncols-1
            shifted_matrix[i, j+1] = matrix[i, j]
        end
        shifted_matrix[i, end] = matrix[i, end]  # Copy the last column
    end

    return shifted_matrix
end


# Creating a gif of the iterations
grid = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
global gr = deepcopy(grid)
@gif for i in 1:400
        global gr
        gr = bacteria_growth_iteration(gr,0.1,i,0.15,0.01)
        h1 = heatmap(gr[1], color = cgrad([:seashell4,:black,:red,:yellow],4,categorical = true), legend=false,yflip = true,categorical = true,clim = (-1,2),title=string("Plot of the states B and I for ", L"p_f^{abs}=0.15,p_c^{abs}=0.01"))
        h2 = heatmap(gr[6],color = :batlowK,legend=true,yflip = true,clim=(-1,20), title= string("Plot of the corresponding environmental AIP concentration ", L"P_E"))
        p = plot(h1,h2,layout=(2,1),size=(700,700))
        display(p)
    end


"""
Counts the number of entries with value "val" in the matrix in the flow channel
"""
function count_induced(matrix,val)
    count = 0
    for i in 2:rows+1
        for element in matrix[i, :]
            if element == val
                count += 1
            end
        end
    end
    return count
end

"""
Counts the number of entries with value "val" in the matrix in the crevices
"""
function count_induced_c(matrix,val)
    count = 0
    for i in rows+2:(4*rows+2)
        for element in matrix[i, :]
            if element == val
                count += 1
            end
        end
    end
    return count
end


# Change of the number of induced bacteria in the flow channel with respect to changes in p_abs_f
function plot_density_induced_flow()
    density_1 = zeros(21)
    density_2 = zeros(21)
    density_3 = zeros(21)
    density_4 = zeros(21)
    density_5 = zeros(21)
    p = collect(0:0.05:1)

    g_0 = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    for j in 1:length(p)
        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,p[j],0.01)
        end
        plot_grid(gr)
        density_1[j] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,p[j],0.01)
        end
        plot_grid(gr)
        density_2[j] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,p[j],0.01)
        end
        plot_grid(gr)
        density_3[j] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,p[j],0.01)
        end
        plot_grid(gr)
        density_4[j] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,p[j],0.01)
        end
        plot_grid(gr)
        density_5[j] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))
    end
    plt = plot(p,density_1,ylims=(0,1))
    plot!(p,density_2,ylims=(0,1))
    plot!(p,density_3,ylims=(0,1))
    plot!(p,density_4,ylims=(0,1))
    plot!(p,density_5,ylims=(0,1))
    display(plt)
    return [density_1,density_2,density_3,density_4,density_5]
end
@time dens_f = plot_density_induced_flow()
plot(collect(0:0.1:1),mean(dens_f),yerror = std(dens),ylims=(0,1))
plot(collect(0:0.05:1),dens_f,ylims=(0,1),label=["Realization 1" "Realization 2" "Realization 3" "Realization 4" "Realization 5"],legend =:bottomright,xaxis=L"p_f^{abs}",yaxis=L"\theta_F(400)")
plot(collect(0:0.05:1),mean(dens_f),ribbon = std(dens_f),ylims=(0,1),label="Mean of five realizations",lw=2,legend =:bottomright,xaxis=L"p_f^{abs}",yaxis=L"\theta_F(400)")
# Fitting the resulting points to a Hill Function
function residuals(p)
    x = collect(0:0.05:1)
    hill(x) = p[3]*x^p[1]/(p[2]^p[1]+x^p[1])
    return mse(hill.(x), mean(dens_f))
end
x0 = [3,0.1,mean(dens_f)[21]]
opt = optimize(residuals,x0,NelderMead())
o = Optim.minimizer(opt)
hill(x) = o[3]*x^o[1]/(o[2]^o[1]+x^o[1])
plot!(collect(0:0.05:1),hill,ylims=(0,1),label="Fitted Hill Function H")


# Change of the number of induced bacteria in the flow channel with respect to changes in p_abs_f but with different intial conditions
function plot_density_induced_flow_init()
    density_1 = zeros(21)
    density_2 = zeros(21)
    density_3 = zeros(21)
    density_4 = zeros(21)
    density_5 = zeros(21)
    p = collect(0:0.05:1)
    for i in 1:length(p)
        grid = cellular_automata(400,p[i],0.01)
        density_1[i] = count_induced(grid[1],2)/(count_induced(grid[1],1)+count_induced(grid[1],2))

        grid = cellular_automata(400,p[i],0.01)
        density_2[i] = count_induced(grid[1],2)/(count_induced(grid[1],1)+count_induced(grid[1],2))
    
        grid = cellular_automata(400,p[i],0.01)
        density_3[i] = count_induced(grid[1],2)/(count_induced(grid[1],1)+count_induced(grid[1],2))
    
        grid = cellular_automata(400,p[i],0.01)
        density_4[i] = count_induced(grid[1],2)/(count_induced(grid[1],1)+count_induced(grid[1],2))
    
        grid = cellular_automata(400,p[i],0.01)
        density_5[i] = count_induced(grid[1],2)/(count_induced(grid[1],1)+count_induced(grid[1],2))
    end
    plt = plot(p,density_1,ylims=(0,1),label=false)
    plot!(p,density_2,ylims=(0,1))
    plot!(p,density_3,ylims=(0,1))
    plot!(p,density_4,ylims=(0,1))
    plot!(p,density_5,ylims=(0,1))
    display(plt)
    return [density_1,density_2,density_3,density_4,density_5]
end
@time dens_init = plot_density_induced_flow_init()


"""
Calculating the larges successive difference in a vector
"""
function max_successive_difference(vector)
    max_diff = -Inf
    max_diff_index = 0
    
    for i in 2:length(vector)
        diff = abs(vector[i] - vector[i-1])
        if diff > max_diff
            max_diff = diff
            max_diff_index = i - 1  # Subtract 1 to get the index of the first element in the pair
        end
    end
    
    return max_diff, max_diff_index
end
# Example usage for a prior calculated density vector
max_diff, max_diff_index = max_successive_difference(mean(dens))


# Change of the number of induced bacteria in the crevices with respect to changes in p_abs_c
function plot_density_induced_flow_c()
    density_1 = zeros(21)
    density_2 = zeros(21)
    density_3 = zeros(21)
    density_4 = zeros(21)
    density_5 = zeros(21)
    p = collect(0:0.005:0.1)

    g_0 = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    for j in 1:length(p)
        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,p[j])
        end
        plot_grid(gr)
        density_1[j] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,p[j])
        end
        plot_grid(gr)
        density_2[j] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,p[j])
        end
        plot_grid(gr)
        density_3[j] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,p[j])
        end
        plot_grid(gr)
        density_4[j] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))

        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,p[j])
        end
        plot_grid(gr)
        density_5[j] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))
    end
    plt = plot(p,density_1,ylims=(0,1))
    plot!(p,density_2,ylims=(0,1))
    plot!(p,density_3,ylims=(0,1))
    plot!(p,density_4,ylims=(0,1))
    plot!(p,density_5,ylims=(0,1))
    display(plt)
    return [density_1,density_2,density_3,density_4,density_5]
end
@time dens = plot_density_induced_flow_c()
plot(collect(0:0.005:0.1),dens,ylims=(0,1),label=["Realization 1" "Realization 2" "Realization 3" "Realization 4" "Realization 5"],legend =:bottomright,xaxis=L"p_c^{abs}",yaxis=L"\theta_C(400)")
plot(collect(0:0.005:0.1),mean(dens),ribbon = std(dens),ylims=(0,1),label="Mean of five realizations",lw=2,legend =:bottomright,xaxis=L"p_c^{abs}",yaxis=L"\theta_C(400)")
# Fitting the resulting points to an exponential growth function with saturation
function residuals(p)
    x = collect(0:0.005:0.1)
    exp_sat(x) = p[1]*(1-exp(-p[2]*x))
    return mse(exp_sat.(x), mean(dens))
end
x0 = [0.8,100.0]
opt = optimize(residuals,x0,NelderMead())
o = Optim.minimizer(opt)
exp_sat(x) = o[1]*(1-exp(-o[2]*x))
plot!(collect(0:0.005:0.1),exp_sat,ylims=(0,1),label="Fitted Growth Function g",lw=2)






# Visualization of the grid G
z = initialize_grid(10,200)
heatmap(z, color = cgrad([:seashell4,:black,:red,:yellow],4,categorical = true), legend=false,yflip = true,categorical = true,clim = (-1,2),size=(2000,1000),grid=false,axis=false)
rows, cols = size(z)
for i in 1:rows+1
    plot!([0.5, cols+0.5], [i-0.5, i-0.5], color=:grey, linewidth=2, legend=false)
end
for j in 1:cols+1
    plot!([j-0.5, j-0.5], [0.5, rows+0.5], color=:grey, linewidth=2, legend=false)
end
display(plot!())



# Visualization of a von neumann neighborhood
function von_neumann_neighborhood(n, r, center)
    x0, y0 = center
    neighborhood = []
    for x in 1:n
        for y in 1:n
            if abs(x - x0) + abs(y - y0) <= r
                push!(neighborhood, (x, y))
            end
        end
    end
    return neighborhood
end
n = 11
w = 3  
center = (6, 6) 
neighborhood = von_neumann_neighborhood(n, w, center)
grid = zeros(Int, n, n)
for (x, y) in neighborhood
    grid[y, x] = 1  # Note that grid indexing is grid[row, col]
    if (x,y) == center
        grid[y,x] = 2
    end
end
hm = heatmap(grid, c=cgrad([:white,:black,:red],4,categorical = true), categorical = true,title="Von Neumann Neighbourhood (r = $w)", legend=false,axis=false)
for i in 0:n
    plot!([i+0.5, i+0.5], [0.5, n+0.5], color=:black, lw=1,legend=false)  # Vertical lines
    plot!([0.5, n+0.5], [i+0.5, i+0.5], color=:black, lw=1,legend=false)  # Horizontal lines
end
display(hm)



# Performing several runs of the CA up to a time point and calculating the ratio of induced bacteria in the crevices 
function stationary_measure(n)
    measure = zeros(n)
    g_0 = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    for j in 1:n
        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,0.01)
        end
        plot_grid(gr)
        # grid = cellular_automata(400,0.15,0.03)
        measure[j] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))
    end
    return measure
end
@time s_measure_c = stationary_measure(100)
d_plot = density(s_measure_c,lw=2,label="Density of Data",size = (900,600),legend =:outertopright,legendfontsize=12,xaxis=L"\theta_C(400)")
# Fitting the data to a normal distribution via a maximum likelihood estimation
fitted_dist = fit(Normal,s_measure_c)
x = range(xlims(d_plot)[1], xlims(d_plot)[2], length=1000)
μ, σ = params(fitted_dist)
label_text = string("Fitted Normal Distribution \n", L"\mathcal{N}(", round(μ, digits=3), ", ", round(σ, digits=3), ")")
plot!(x, pdf.(fitted_dist, x), label=label_text, lw=2)
# Performing a Kolmogorov-Smirnov Test for the fitted normal distribution and the data
HypothesisTests.ExactOneSampleKSTest(s_measure_c,fitted_dist)


# Performing several runs of the CA up to a time point and calculating the ratio of induced bacteria in the flow channel 
function stationary_measure_f(n)
    measure = zeros(n)
    g_0 = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    for j in 1:n
        gr = deepcopy(g_0)
        for i in 1:400
            gr = bacteria_growth_iteration(gr,0.1,i,0.1,0.01)
        end
        plot_grid(gr)
        # grid = cellular_automata(400,0.15,0.03)
        measure[j] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))
    end
    return measure
end
@time s_measure = stationary_measure_f(100)
d_plot = density(s_measure,lw=2,label="Density of Data",size = (900,600),legend =:outertopright,legendfontsize=12,xaxis=L"\theta_F(400)")
# Fitting the data to a normal distribution via a maximum likelihood estimation
fitted_dist = fit(Normal,s_measure)
x = range(xlims(d_plot)[1], xlims(d_plot)[2], length=1000)
μ, σ = params(fitted_dist)
label_text = string("Fitted Normal Distribution \n", L"\mathcal{N}(", round(μ, digits=3), ", ", round(σ, digits=3), ")")
plot!(x, pdf.(fitted_dist, x), label=label_text, lw=2)
# Performing a Kolmogorov-Smirnov Test for the fitted normal distribution and the data
HypothesisTests.ExactOneSampleKSTest(s_measure,fitted_dist)


# Performing several runs of the CA up to a time point with varying initial conditions and calculating the ratio of induced bacteria in the flow channel
function stationary_measure_f_init(n)
    measure = zeros(n)
    for j in 1:n
        grid = cellular_automata(400,0.1,0.01)
        measure[j] = count_induced_c(grid[1],2)/(count_induced_c(grid[1],1)+count_induced_c(grid[1],2))
    end
    return measure
end
@time s_measure = stationary_measure_f_init(20)
d_plot = density(s_measure,lw=2,label="Density of Data",size = (900,600),legend =:outertopright,legendfontsize=12)
fitted_dist = fit(Normal,s_measure)
x = range(xlims(d_plot)[1], xlims(d_plot)[2], length=1000)
μ, σ = params(fitted_dist)
label_text = string("Fitted Normal Distribution \n", L"\mathcal{N}(", round(μ, digits=3), ", ", round(σ, digits=3), ")")
plot!(x, pdf.(fitted_dist, x), label=label_text, lw=2)
HypothesisTests.ExactOneSampleKSTest(s_measure,fitted_dist)


"""
Calculating the ratio of induced bacteria to basal bacteria in the crevices for each step of the CA until timepoint t
"""
function θ_C(t)
    evolution = zeros(t)
    g_0 = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    gr = deepcopy(g_0)
    for i in 1:t
        gr = bacteria_growth_iteration(gr,0.1,i,0.15,0.01)
        evolution[i] = count_induced_c(gr[1],2)/(count_induced_c(gr[1],1)+count_induced_c(gr[1],2))
    end
    plot_grid(gr)

    return evolution
end
@time evo = θ_C(1200)
plot(evo,label=L"\theta_C(t)",xaxis="t")
# Fitting the resulting points to a Hill Function 
function residuals_θ(p)
    x = collect(1:1:1200)
    hill(x) = p[3]*x^p[1]/(p[2]^p[1]+x^p[1])
    return mse(hill.(x), evo)
end
x0 = [3,400.0,1]
opt = optimize(residuals_θ,x0,NelderMead())
o = Optim.minimizer(opt)
hill(x) = o[3]*x^o[1]/(o[2]^o[1]+x^o[1])
plot!(collect(1:1:1200),hill,ylims=(0,1),label=string("Fitted Hill Function ",L"H_C(t)"))



"""
Calculating the ratio of induced bacteria to basal bacteria in the flow channel for each step of the CA until timepoint t
"""
function θ_F(t)
    evolution = zeros(t)
    g_0 = [initialize_grid(rows, cols),initialize_AIP(rows,cols),initialize_nutrients(rows,cols),initialize_nutrients_flow(rows,cols),initialize_AIP(rows, cols),initialize_AIP(rows, cols)]
    gr = deepcopy(g_0)
    for i in 1:t
        gr = bacteria_growth_iteration(gr,0.1,i,0.15,0.01)
        evolution[i] = count_induced(gr[1],2)/(count_induced(gr[1],1)+count_induced(gr[1],2))
    end
    plot_grid(gr)

    return evolution
end
@time evo_2 = θ_F(1200)
plot(evo_2,label=L"\theta_F(t)",axis="t")
# Fitting the resulting points to a Hill Function
function residuals_θ_F(p)
    x = collect(1:1:1200)
    hill(x) = p[3]*x^p[1]/(p[2]^p[1]+x^p[1])
    return mse(hill.(x), evo_2)
end
x0 = [3,300.0,1]
opt = optimize(residuals_θ_F,x0,NelderMead())
o = Optim.minimizer(opt)
hill(x) = o[3]*x^o[1]/(o[2]^o[1]+x^o[1])
plot!(collect(1:1:1200),hill,label=string("Fitted Hill Function ",L"H_F(t)"),legend=:bottomright)



end