module Quorum_Sensing

using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
Pkg.add("IntervalArithmetic")
Pkg.add("IntervalRootFinding")
Pkg.add("MTH229")
Pkg.add("DynamicalSystems")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DiffEqParamEstim")
Pkg.add("Optimization")
Pkg.add("Metrics")
Pkg.add("Optim")
Pkg.add("OptimizationOptimJL")
Pkg.add("BifurcationKit")
Pkg.add("Parameters")
Pkg.add("Revise")
Pkg.add("NLsolve")
Pkg.add("StaticArrays")
Pkg.add("DDEBifurcationKit")
Pkg.add("Distributions")
Pkg.add("ColorSchemes")
Pkg.add("Colors")
Pkg.add("LaTeXStrings")
Pkg.add("StatsPlots")
Pkg.add("Distributions")
Pkg.add("HypothesisTests")

using DifferentialEquations
using Plots
using IntervalArithmetic
using IntervalRootFinding
using MTH229
using DynamicalSystems
using CSV
using DataFrames
using DiffEqParamEstim
using Optimization
using Metrics
using Optim
using OptimizationOptimJL
using BifurcationKit
using Parameters
using Revise
using NLsolve
using StaticArrays
using DDEBifurcationKit
using Distributions
using Random
using ColorSchemes
using Colors
using LaTeXStrings
using StatsPlots
using Distributions
using HypothesisTests


# -----------------------------------------------------------------------------------
# THE ORIGINAL MODEL


"""
The Original Model
"""
function qs(du,u,p,t)
    AgrA,AgrB,AgrC,AgrD,AIP,AgrC_AIP,AgrAP = u
    α_A,α_B,α_C,α_D,
    β_A,β_B,β_C,β_D,
    γ_A,γ_B,γ_C,γ_D,γ_AP,γ_AIP,γ_CAIP,
    K_AgrAP,K_C,K_F,K_DP,K_m,K_P,
    v_max = p
    du[1] = (α_A+β_A*AgrAP/(K_AgrAP+AgrAP))-γ_A*AgrA-K_P*AgrC_AIP*AgrA+K_DP*AgrAP
    du[2] = (α_B+β_B*AgrAP/(K_AgrAP+AgrAP))-γ_B*AgrB
    du[3] = (α_C+β_C+AgrAP/(K_AgrAP+AgrAP))-γ_C*AgrC-K_C*AIP*AgrC+K_F*AgrC_AIP
    du[4] = (α_D+β_D*AgrAP/(K_AgrAP+AgrAP))-γ_D*AgrD-v_max*AgrD/(K_m+AgrD)
    du[5] = v_max*AgrD/(K_m+AgrD)-γ_AIP*AIP-K_C*AIP*AgrC+K_F*AgrC_AIP
    du[6] = -γ_CAIP*AgrC_AIP+K_C*AIP*AgrC-K_F*AgrC_AIP
    du[7] = K_P*AgrC_AIP*AgrA-K_DP*AgrAP-γ_AP*AgrAP
end
# Example for the original model 
tspan = (0.0,10.0)
init = 0.1*ones(7)
parms = rand(22)
qs_prob = ODEProblem(qs,init,tspan,parms)
qs_sol = solve(qs_prob,Tsit5(),saveat = 0.1)
p = plot(qs_sol,label = ["AgrA" "AgrB" "AgrC" "AgrD" "AIP" "AgrCAIP" "AgrAP"],linewidth = 2,legend = :outertopright,dpi = 300)



ϑ(AgrAP,K_AgrAP) = AgrAP/(K_AgrAP+AgrAP)        # Function ϑ as defined in the bachelor thesis


# -----------------------------------------------------------------------------------
# CONTOUR PLOTS

# Example for a contour plot 
γ_A = 1
γ_AP = 1
γ_CAIP = 1
γ_C = 1
K_C = 1
α_C = 1
β_C = 1
K_AgrAP = 1
K_P = 1
α_A = 1
β_A = 1
f(AgrAP,AIP) = AgrAP*(γ_A*γ_AP*γ_CAIP*(γ_C+K_C*AIP)+γ_AP*α_C*K_C*AIP + γ_AP*β_C*ϑ(AgrAP,K_AgrAP)*K_C*AIP) - K_P*K_C*AIP*(α_A*α_C + α_C*β_A*ϑ(AgrAP,K_AgrAP)+α_A*β_C*ϑ(AgrAP,K_AgrAP)+β_A*β_C*ϑ(AgrAP,K_AgrAP)^2)
AIP = range(0, 100, length=100)
AgrAP = range(0, 100, length=100)
contour(AIP,AgrAP,f,color=:turbo,clabels=true,cbar = false,levels = [0])

"""
Countour plot for f with variable n corresponding to vayring one parameter
"""
function change_f(n::Real)
    γ_A = ones(100)
    γ_AP = ones(100)
    γ_CAIP = ones(100)
    γ_C = ones(100)
    K_C = ones(100)
    α_C = ones(100)
    β_C = ones(100)
    K_AgrAP = ones(100)
    K_P = ones(100)
    α_A = ones(100)
    β_A = ones(100)
    if n == 1
        γ_A = range(0,10,length=100)
    elseif n == 2
        γ_AP = range(0,10,length=100)
    elseif n == 3
        γ_CAIP = range(0,10,length=100)
    elseif n == 4
        γ_C = range(0,10,length=100)
    elseif n == 5
        K_C = range(0,10,length=100)
    elseif n == 6
        α_C = range(0,10,length=100)
    elseif n == 7
        β_C = range(0,10,length=100)
    elseif n == 8
        K_AgrAP = range(0,10,length=100)
    elseif n == 9
        K_P = range(0,10,length=100)
    elseif n == 10
        α_A = range(0,10,length=100)
    elseif n == 11
        β_A = range(0,10,length=100)
    end

    for i = 1:100
        f(AgrAP,AIP) = AgrAP*(γ_A[i]*γ_AP[i]*γ_CAIP[i]*(γ_C[i]+K_C[i]*AIP)+γ_AP[i]*α_C[i]*K_C[i]*AIP + γ_AP[i]*β_C[i]*ϑ(AgrAP,K_AgrAP[i])*K_C[i]*AIP) - K_P[i]*K_C[i]*AIP*(α_A[i]*α_C[i] + α_C[i]*β_A[i]*ϑ(AgrAP,K_AgrAP[i])+α_A[i]*β_C[i]*ϑ(AgrAP,K_AgrAP[i])+β_A[i]*β_C[i]*ϑ(AgrAP,K_AgrAP[i])^2)
        # AIP = range(0, 10, length=100)
        # AgrAP = range(0, 1000, length=1000)
        AIP = range(0, 2, length=100)
        AgrAP = range(0, 15, length=100)
        if i == 1
            global c = contour(AIP,AgrAP,f,color=:blue,cbar = false,levels = [0])
        else
            contour!(AIP,AgrAP,f,color=:blue,cbar = false,levels = [0])
        end

        g(AgrAP,AIP) = AIP-500*AgrAP^2
        contour!(AIP,AgrAP,g,color=:red,cbar = false,levels = [0],lw=2)

        q(AgrAP,AIP) = AIP-2*AgrAP^2
        contour!(AIP,AgrAP,q,color=:red,cbar = false,levels = [0],lw=2)


        if i == 100
            xlabel!("AIP")
            ylabel!("AgrAP")
            display(c)
        end
    end
end


# Example for a contour plot
vmax = 1
K_m = 1
α_D = 1
β_D = 1
K_AgrAP = 1
γ_D = 1
f(AgrAP,AgrD) = vmax*AgrD/(K_m+AgrD) - α_D - β_D*AgrAP/(K_AgrAP+AgrAP) + γ_D*AgrD
AgrD = range(0, 10, length=1000)
AgrAP = range(0, 100, length=1000)
contour(AgrAP,AgrD,f,color=:turbo,clabels=true,cbar = false,levels = [0])

"""
Countour plot for g with variable n corresponding to vayring one parameter
"""
function change_g(n::Real)
    vmax = ones(100)
    K_m = ones(100)
    α_D = ones(100)
    β_D = ones(100)
    K_AgrAP = ones(100)
    γ_D = ones(100)
    if n == 1
        vmax = range(0,10,length=100)
    elseif n == 2
        K_m = range(0,10,length=100)
    elseif n == 3
        α_D = range(0,10,length=100)
    elseif n == 4
        β_D = range(0,10,length=100)
    elseif n == 5
        K_AgrAP = range(0,10,length=100)
    elseif n == 6
        γ_D = range(0,10,length=100)
    end

    for i = 1:100
        f(AgrAP,AgrD) = vmax[i]*AgrD/(K_m[i]+AgrD) - α_D[i] - β_D[i]*AgrAP/(K_AgrAP[i]+AgrAP) + γ_D[i]*AgrD
        # AIP = range(0, 10, length=100)
        # AgrAP = range(0, 1000, length=1000)
        AgrD = range(0, 2, length=100)
        AgrAP = range(0, 10, length=100)
        if i == 1
            global c = contour(AgrAP,AgrD,f,color=:turbo,cbar = false,levels = [0])
        else
            contour!(AgrAP,AgrD,f,color=:turbo,clabels = true, cbar = false,levels = [0])
        end

        λ = (+sqrt(5)+5)/(1+2*sqrt(2)-sqrt(5))*2/5
        # λ = 1
        g(AgrAP,AgrD) = (2*sqrt(2)+1-sqrt(5))/(λ*AgrAP+2)+AgrD-sqrt(2)
        contour!(AgrAP,AgrD,g,color=:red,cbar = false,levels = [0])

        if i == 100
            xlabel!("AgrAP")
            ylabel!("AgrD")
            display(c)
        end
    end

    r(AgrAP) = (-(α_D[1]+β_D[1]*AgrAP/(K_AgrAP[1]+AgrAP)-γ_D[1]*K_m[1]-vmax[1])-sqrt((α_D[1]+β_D[1]*AgrAP/(K_AgrAP[1]+AgrAP)-γ_D[1]*K_m[1]-vmax[1])^2+4*γ_D[1]*(α_D[1]*K_m[1]+β_D[1]*AgrAP/(K_AgrAP[1]+AgrAP)*K_m[1])))*1/(-2*γ_D[1])
    #p = plot!(r,0,10,label = ["Root function" "contour"])
    # scatter!([5],[r(5)])
end

"""
Comparing the function g with its approximation
"""
function comparison(vmax,K_m,α_D,β_D,K_AgrAP,γ_D)
    g(x) = ((α_D+β_D*ϑ(x,K_AgrAP)-γ_D*K_m-vmax)+sqrt((α_D+β_D*ϑ(x,K_AgrAP)-γ_D*K_m-vmax)^2+4*γ_D*(α_D*K_m+β_D*ϑ(x,K_AgrAP)*K_m)))/(2*γ_D)
    q(x) = -K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*x+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax)
    p = plot(g,0,100,label = L"g")
    plot!(q,0,100,label = "Approximation",xaxis="[AgrAP]")
    display(p)
    c = range(0,10,100000)
    return mse(g.(c),q.(c))
end

comparison(5,0.01,1,20,1,1)



# -----------------------------------------------------------------------------------
# SIMPLIFIED MODEL

"""
Bifurcation of the simplified one-dimensional model
"""
function bifurcation(K_C,α_C,β_C,γ_C,K_AgrAP,α_D,β_D,γ_D,γ_AIP,K_m,vmax,κ)
    r(β_D,α_D,γ_D,K_m,vmax) = sqrt((α_D+β_D-γ_D*K_m-vmax)^2+4*γ_D*(α_D*K_m+β_D*K_m))
    K_1(β_D,α_D,γ_D,K_m,vmax) = β_D+r(β_D,α_D,γ_D,K_m,vmax)-r(0,α_D,γ_D,K_m,vmax)
    K_2(γ_D) = 2*γ_D
    K_3(β_D,α_D,γ_D,K_m,vmax) = (α_D+β_D-γ_D*K_m-vmax+r(β_D,α_D,γ_D,K_m,vmax))/(2*γ_D)
    λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) = (2*β_D*γ_D)/(K_AgrAP*(β_D+r(β_D,α_D,γ_D,K_m,vmax)-r(0,α_D,γ_D,K_m,vmax)))*((α_D+K_m*γ_D-vmax)/(r(0,α_D,γ_D,K_m,vmax))+1)

    f(x) = K_C*x*(α_C+β_C*x^2/(K_AgrAP+x^2))/(γ_C+K_C*x)+γ_AIP*x
    g(x) = α_D+β_D*κ*x^2/(K_AgrAP+κ*x^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*x^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))
    h(x) = α_D+β_D*κ*x^2/(K_AgrAP+κ*x^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*x^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-γ_AIP*x - K_C*x*(α_C+β_C*x^2/(K_AgrAP+x^2))/(γ_C+K_C*x)
    
    p = plot(g,0,4,label = L"p_1",xaxis="x")
    plot!(f,0,4,label= L"p_2, K_C = 1")
    q(x) = (K_C*γ_AIP*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*x^6 
            + (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + β_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*x^5 
            + (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP*κ + K_AgrAP*K_C*γ_AIP*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*x^4 
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ + γ_AIP*γ_C*κ + (α_C*κ - α_D*κ + β_C*κ - β_D*κ)*K_C)*K_2(γ_D))*x^3  
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP - α_D*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - α_D*γ_C*κ - β_D*γ_C*κ)*K_2(γ_D))*x^2 
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D + K_C*(α_C - α_D) + γ_AIP*γ_C)*K_2(γ_D))*K_AgrAP*x
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - α_D*γ_C)*K_2(γ_D))*K_AgrAP)
    
    
    I = IntervalRootFinding.roots(h, 0..10)
    m = mid.(interval.(I))
    scatter!(m,f.(m),label = false,color = :green)


    K_C = 5
    plot!(f,0,4, label= L"p_2, K_C = 5")
    I = IntervalRootFinding.roots(h, 0..10)
    m = mid.(interval.(I))
    scatter!(m,f.(m),label = false,color = [:red,:green,:green])

    K_C = 40
    plot!(f,0,4, label= L"p_2, K_C = 40")
    I = IntervalRootFinding.roots(h, 0..10)
    m = mid.(interval.(I))
    scatter!(m,f.(m),label = "Intersections",color = :green)


    display(p)
end

# Example for a bifurcation
bifurcation(1,1.8,0.01,2,2,0.8,5.5,0.1,1,2,3,1)


"""
Display the coefficients of the polynomial that determines the roots of p
"""
function coefficients(K_C,α_C,β_C,γ_C,K_AgrAP,α_D,β_D,γ_D,γ_AIP,K_m,vmax,κ)
    q(x) = (K_C*γ_AIP*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*x^6 
            + (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + β_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*x^5 
            + (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP*κ + K_AgrAP*K_C*γ_AIP*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*x^4 
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ + γ_AIP*γ_C*κ + (α_C*κ - α_D*κ + β_C*κ - β_D*κ)*K_C)*K_2(γ_D))*x^3  
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP - α_D*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - α_D*γ_C*κ - β_D*γ_C*κ)*K_2(γ_D))*x^2 
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D + K_C*(α_C - α_D) + γ_AIP*γ_C)*K_2(γ_D))*K_AgrAP*x
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - α_D*γ_C)*K_2(γ_D))*K_AgrAP)

    display(K_C*γ_AIP*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))
    display(+(K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + β_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C))
    display(+(K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP*κ + K_AgrAP*K_C*γ_AIP*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)))
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ + γ_AIP*γ_C*κ + (α_C*κ - α_D*κ + β_C*κ - β_D*κ)*K_C)*K_2(γ_D)))
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP - α_D*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - α_D*γ_C*κ - β_D*γ_C*κ)*K_2(γ_D)))
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D + K_C*(α_C - α_D) + γ_AIP*γ_C)*K_2(γ_D))*K_AgrAP)
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - α_D*γ_C)*K_2(γ_D))*K_AgrAP)
end

# Example
coefficients(1,1.8,0.01,2,2,0.8,5.5,0.1,1,2,3,1)

"""
Visual determination of the stability of the simplified one-dimensional model
"""
function Stability(K_C,α_C,β_C,γ_C,K_AgrAP,α_D,β_D,γ_D,γ_AIP,K_m,vmax,κ)
    r(β_D,α_D,γ_D,K_m,vmax) = sqrt((α_D+β_D-γ_D*K_m-vmax)^2+4*γ_D*(α_D*K_m+β_D*K_m))
    K_1(β_D,α_D,γ_D,K_m,vmax) = β_D+r(β_D,α_D,γ_D,K_m,vmax)-r(0,α_D,γ_D,K_m,vmax)
    K_2(γ_D) = 2*γ_D
    K_3(β_D,α_D,γ_D,K_m,vmax) = (α_D+β_D-γ_D*K_m-vmax+r(β_D,α_D,γ_D,K_m,vmax))/(2*γ_D)
    λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) = (2*β_D*γ_D)/(K_AgrAP*(β_D+r(β_D,α_D,γ_D,K_m,vmax)-r(0,α_D,γ_D,K_m,vmax)))*((α_D+K_m*γ_D-vmax)/(r(0,α_D,γ_D,K_m,vmax))+1)

    f(x) = K_C*x*(α_C+β_C*x^2/(K_AgrAP+x^2))/(γ_C+K_C*x)+γ_AIP*x
    g(x) = α_D+β_D*κ*x^2/(K_AgrAP+κ*x^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*x^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))
    h(x) = α_D+β_D*κ*x^2/(K_AgrAP+κ*x^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*x^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-γ_AIP*x - K_C*x*(α_C+β_C*x^2/(K_AgrAP+x^2))/(γ_C+K_C*x)
    
    p = plot(h,0,4,label = L"p",xaxis="x")

    q(x) = (K_C*γ_AIP*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*x^6 
            + (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + β_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*x^5 
            + (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP*κ + K_AgrAP*K_C*γ_AIP*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*x^4 
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ + γ_AIP*γ_C*κ + (α_C*κ - α_D*κ + β_C*κ - β_D*κ)*K_C)*K_2(γ_D))*x^3  
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP - α_D*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - α_D*γ_C*κ - β_D*γ_C*κ)*K_2(γ_D))*x^2 
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D + K_C*(α_C - α_D) + γ_AIP*γ_C)*K_2(γ_D))*K_AgrAP*x
            - (K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - α_D*γ_C)*K_2(γ_D))*K_AgrAP)
    

    I = IntervalRootFinding.roots(h, 0..10)
    m = mid.(interval.(I))
    w(x) = 0
    plot!(w,0,4,linestyle=:dash,label = false,color=:orange)
    scatter!(m,h.(m),label = false,color = [:red,:green,:green])
    quiver!([0.6],[0.0],quiver=([-0.1],[0.0]),color=:orange)
    quiver!([0.0],[0.0],quiver=([0.1],[0.0]),color=:orange)
    quiver!([1.5],[0.0],quiver=([0.1],[0.0]),color=:orange)
    quiver!([2.7],[0.0],quiver=([-0.1],[0.0]),color=:orange)
    display(p)


    display(K_C*γ_AIP*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))
    display(+(K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + β_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C))
    display(+(K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP*κ + K_AgrAP*K_C*γ_AIP*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - β_D*γ_C*κ*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)))
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + γ_AIP*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + (α_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) - α_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_C)*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D*κ + γ_AIP*γ_C*κ + (α_C*κ - α_D*κ + β_C*κ - β_D*κ)*K_C)*K_2(γ_D)))
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) + K_2(γ_D)*K_C*γ_AIP - α_D*γ_C*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP))*K_AgrAP - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D*κ - α_D*γ_C*κ - β_D*γ_C*κ)*K_2(γ_D)))
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*K_C*γ_D + K_C*(α_C - α_D) + γ_AIP*γ_C)*K_2(γ_D))*K_AgrAP)
    display(-(K_1(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - (K_3(β_D,α_D,γ_D,K_m,vmax)*γ_C*γ_D - α_D*γ_C)*K_2(γ_D))*K_AgrAP)
end

# Example
Stability(5,1.8,0.01,2,2,0.8,5.5,0.1,1,2,3,1)


# Defintion of the parameters used for the approximation of g
r(β_D,α_D,γ_D,K_m,vmax) = sqrt((α_D+β_D-γ_D*K_m-vmax)^2+4*γ_D*(α_D*K_m+β_D*K_m))
K_1(β_D,α_D,γ_D,K_m,vmax) = β_D+r(β_D,α_D,γ_D,K_m,vmax)-r(0,α_D,γ_D,K_m,vmax)
K_2(γ_D) = 2*γ_D
K_3(β_D,α_D,γ_D,K_m,vmax) = (α_D+β_D-γ_D*K_m-vmax+r(β_D,α_D,γ_D,K_m,vmax))/(2*γ_D)
λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP) = (2*β_D*γ_D)/(K_AgrAP*(β_D+r(β_D,α_D,γ_D,K_m,vmax)-r(0,α_D,γ_D,K_m,vmax)))*((α_D+K_m*γ_D-vmax)/(r(0,α_D,γ_D,K_m,vmax))+1)


# p[1] = K_C
# p[2] = α_C
# p[3] = β_C
# p[4] = γ_C
# p[5] = K_AgrAP
# p[6] = α_D
# p[7] = β_D
# p[8] = γ_D
# p[9] = γ_AIP
# p[10] = K_m
# p[11] = vmax
# p[12] = κ
"""
ODE of the simplified model
"""
solo_AIP(x,p,t) = p[6].+p[7].*p[12].*x.^2 ./(p[5].+p[12].*x.^2).-p[8].*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5]).*p[12].*x.^2 .+K_2(p[8])).+K_3(p[7],p[6],p[8],p[10],p[11])).-p[9].*x .- p[1].*x.*(p[2].+p[3]*x.^2 ./(p[5].+x.^2))./(p[4].+p[1].*x)
tspan = (0.0,20.0)
init = 1.0
parms = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.,1.,1.,1.,1.]
AIP_prob = ODEProblem{false}(solo_AIP,init,tspan,parms)
AIP_sol = solve(AIP_prob,saveat = 0.1,alg=Tsit5()) # Oscilations for Tsit5 solver!
p = plot(AIP_sol,linewidth = 2,dpi = 300,yaxis="AIP(t)",label=false)



# -----------------------------------------------------------------------------------
# ODE Model for the paper 


"""
ODE Model including flow and topography 
"""
function sqs_ft(du,u,p,t)
    AIP_f,AIP_C1,AIP_C2,N_f,N_C1,N_C2 = u
    K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, r, K_f,K_C1,K_C2, δ, μ, ν, η, σ, ρ, κ = p
    du[1] = (α_D+β_D*κ*AIP_f^2/(K_AgrAP+κ*AIP_f^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f*(α_C+β_C*AIP_f^2/(K_AgrAP+AIP_f^2))/(γ_C+K_C*AIP_f))*N_f - γ_AIP*AIP_f - η*AIP_f - σ*AIP_f + ρ*AIP_C1 - σ*AIP_f + ρ*AIP_C2
    du[2] = (α_D+β_D*κ*AIP_C1^2/(K_AgrAP+κ*AIP_C1^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C1*(α_C+β_C*AIP_C1^2/(K_AgrAP+AIP_C1^2))/(γ_C+K_C*AIP_C1))*N_C1 -γ_AIP*AIP_C1 + σ*AIP_f - ρ*AIP_C1
    du[3] = (α_D+β_D*κ*AIP_C2^2/(K_AgrAP+κ*AIP_C2^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C2*(α_C+β_C*AIP_C2^2/(K_AgrAP+AIP_C2^2))/(γ_C+K_C*AIP_C2))*N_C2 -γ_AIP*AIP_C2 + σ*AIP_f - ρ*AIP_C2
    du[4] = (r+δ)*N_f*(1-N_f/K_f)-μ*N_f+ν*N_C1-μ*N_f+ν*N_C2
    du[5] = r*N_C1*(1-N_C1/K_C1)+μ*N_f-ν*N_C1
    du[6] = r*N_C2*(1-N_C2/K_C2)+μ*N_f-ν*N_C2
end
tspan = (0.0,50.0)
init = [0,0,0,.01,.01,.01]
parms = [.1,1.,1.,1.,1.,1.,.1,0.77,0.01,1.,1.,1.,.1,1.,.01,.1,.01,.01,.01,.01,.01,1.]
# Some other parameter combinations 
# parms = [0.47, 0.87, 0.34, 0.4, 0.74, 0.39, 0.42, 0.71, 0.47, 0.22, 0.11, 0.08, 0.45, 0.01, 0.84, 0.64, 0.61, 0.76, 0.1, 0.72, 0.73,1.]
# parms = [0.87, 0.03, 0.57, 0.38, 0.98, 0.49, 0.27, 0.68, 0.08, 0.97, 0.9, 0.46, 0.79, 0.01, 0.74, 0.37, 0.08, 0.96, 0.78, 0.74, 0.44, 0.37,1.]
# parms = [0.79, 0.8, 0.1, 0.89, 0.79, 0.9, 0.04, 0.06, 0.08, 0.63, 0.6, 0.69, 0.48, 0.22, 1.0, 0.78, 0.57, 0.32, 0.99, 0.81, 0.72, 0.47,1.]
sqs_ft_prob = ODEProblem(sqs_ft,init,tspan,parms)
sqs_ft_sol = solve(sqs_ft_prob,Rosenbrock23())
p = plot(sqs_ft_sol,label = [L"AIP_f" L"AIP_{C_1}" L"AIP_{C_2}" L"N_f" L"N_{C_1}" L"N_{C_2}"],linewidth = 2,legend = :outertopright,dpi = 300)
# Three dimensional plot of the evolution of AIP
plot(sqs_ft_sol[1,:],sqs_ft_sol[2,:],sqs_ft_sol[3,:],xaxis = "AIP_f", yaxis = "AIP_C1", zaxis = "AIP_C2")


# Numerical Solver for the Equilibria. It finds one root. 
function qs_ft(x)
    [(p[6]+p[7]*p[22]*x[1]^2/(p[5]+p[22]*x[1]^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x[1]^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x[1]*(p[2]+p[3]*x[1]^2/(p[5]+x[1]^2))/(p[4]+p[1]*x[1]))*x[4]-p[9]*x[1]-p[19]*x[1]-p[20]*x[1]+p[21]*x[2]-p[20]*x[1]+p[21]*x[3],
    (p[6]+p[7]*p[22]*x[2]^2/(p[5]+p[22]*x[2]^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x[2]^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x[2]*(p[2]+p[3]*x[2]^2/(p[5]+x[2]^2))/(p[4]+p[1]*x[2]))*x[5]-p[9]*x[2]+p[20]*x[1]-p[21]*x[2],
    (p[6]+p[7]*p[22]*x[3]^2/(p[5]+p[22]*x[3]^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x[3]^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x[3]*(p[2]+p[3]*x[3]^2/(p[5]+x[3]^2))/(p[4]+p[1]*x[3]))*x[6]-p[9]*x[3]+p[20]*x[1]-p[21]*x[3],
    (p[12]+p[16])*x[4]*(1-x[4]/p[13])-p[17]*x[4]+p[18]*x[5]-p[17]*x[4]+p[18]*x[6],
    p[12]*x[6]*(1-x[5]/p[14])+p[17]*x[4]-p[18]*x[5],
    p[12]*x[6]*(1-x[6]/p[15])+p[17]*x[4]-p[18]*x[6]]
end
p = [1.,1.,1.,1.,1.,1.,10.,0.77,0.01,1.,1.,1.,.01,1.,1.,.1,.01,.01,9.,.01,.01,1.]
sol = nlsolve(qs_ft, [10., 10., 10., 10., 10., 10.])
sol.zero


# Numerical Solver for the Equilibria. It finds all roots inside the given interval box.
qs_ft( (x1,x2,x3,x4,x5,x6) ) = SVector((p[6]+p[7]*p[22]*x1^2/(p[5]+p[22]*x1^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x1^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x1*(p[2]+p[3]*x1^2/(p[5]+x1^2))/(p[4]+p[1]*x1))*x4-p[9]*x1-p[19]*x1-p[20]*x1+p[21]*x2-p[20]*x1+p[21]*x3,
(p[6]+p[7]*p[22]*x2^2/(p[5]+p[22]*x2^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x2^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x2*(p[2]+p[3]*x2^2/(p[5]+x2^2))/(p[4]+p[1]*x2))*x5-p[9]*x2+p[20]*x1-p[21]*x2,
(p[6]+p[7]*p[22]*x3^2/(p[5]+p[22]*x3^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x3^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x3*(p[2]+p[3]*x3^2/(p[5]+x3^2))/(p[4]+p[1]*x3))*x6-p[9]*x3+p[20]*x1-p[21]*x3,
(p[12]+p[16])*x4*(1-x4/p[13])-p[17]*x4+p[18]*x5-p[17]*x4+p[18]*x6,
p[12]*x6*(1-x5/p[14])+p[17]*x4-p[18]*x5,
p[12]*x6*(1-x6/p[15])+p[17]*x4-p[18]*x6)
p = [5., 1.8, 0.01, 2.0, 2.0, 0.8, 5.5, 0.1, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, .1, 1.0, 1.0,1.]
X = IntervalBox(0..200,0..200,0..200,0..10,0..10,0..10)
@time IntervalRootFinding.roots(qs_ft, X)



# Alternative Model via the substrate concentration
function sqs_ft_S(du,u,p,t)
    AIP_f,AIP_C1,AIP_C2,S_f,S_C1,S_C2,N_f,N_C1,N_C2 = u
    K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, μ, ν, η, σ, ρ, D, a_S, σ_S, ρ_S, ρ_S, a_N, K_S = p
    du[1] = (α_D+β_D*AIP_f^2/(K_AgrAP+AIP_f^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*AIP_f^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f*(α_C+β_C*AIP_f^2/(K_AgrAP+AIP_f^2))/(γ_C+K_C*AIP_f))*N_f - γ_AIP*AIP_f - η*AIP_f - σ*AIP_f + ρ*AIP_C1 - σ*AIP_f + ρ*AIP_C2
    du[2] = (α_D+β_D*AIP_C1^2/(K_AgrAP+AIP_C1^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*AIP_C1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C1*(α_C+β_C*AIP_C1^2/(K_AgrAP+AIP_C1^2))/(γ_C+K_C*AIP_C1))*N_C1 -γ_AIP*AIP_C1 + σ*AIP_f - ρ*AIP_C1
    du[3] = (α_D+β_D*AIP_C2^2/(K_AgrAP+AIP_C2^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*AIP_C2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C2*(α_C+β_C*AIP_C2^2/(K_AgrAP+AIP_C2^2))/(γ_C+K_C*AIP_C2))*N_C2 -γ_AIP*AIP_C2 + σ*AIP_f - ρ*AIP_C2
    du[4] = D-a_S*S_f/(K_S+S_f)*N_f-D*S_f-σ_S*S_f+ρ_S*S_C1-σ_S*S_f+ρ_S*S_C2
    du[5] = -a_S*S_C1/(K_S+S_C1)*N_C1+σ_S*S_f-ρ_S*S_C1
    du[6] = -a_S*S_C2/(K_S+S_C2)*N_C2+σ_S*S_f-ρ_S*S_C2
    du[7] = a_N*S_f/(K_S+S_f)-μ*N_f+ν*N_C1-μ*N_f+ν*N_C2
    du[8] = a_N*S_C1/(K_S+S_C1)+μ*N_f-ν*N_C1
    du[9] = a_N*S_C2/(K_S+S_C2)+μ*N_f-ν*N_C2
end
tspan = (0.0,8.0)
init = [0,0,0,1,1,1,0.1,0.1,0.1]
parms = [1,1,1,1,1,1,10,0.77,1,1,1,1,10,1,10,1,0.1,1,1,1,1,0.01,10]
sqs_ft_prob = ODEProblem(sqs_ft_S,init,tspan,parms)
sqs_ft_sol = solve(sqs_ft_prob,Tsit5())
p = plot(sqs_ft_sol,label = ["AIP_f" "AIP_C1" "AIP_C2" "S_f" "S_C1" "S_C2" "N_f" "N_C1" "N_C2"],linewidth = 2,legend = :outertopright,dpi = 300)




# -----------------------------------------------------------------------------------
# Data


# Reading and plotting the data 
df = CSV.read("Cells_no_flow.csv",DataFrame,header = false, delim = ";",decimal=',',types = [Float64,Float64])
m = Matrix(df)
x = m[:,1]
y = m[:,2]
scatter(x,y,color=:red,label = "No Flow",xaxis="Time (h)",yaxis="Relative cell number",title="Abstracted Data from the paper")

df_2 = CSV.read("Cells_flow.csv",DataFrame,header = false, delim = ";",decimal=',',types = [Float64,Float64])
m_2 = Matrix(df_2)
x_2 = m_2[:,1]
y_2 = m_2[:,2]
scatter!(x_2,y_2,color=:blue,label = "0.5 Flow",xaxis="Time (h)",yaxis="Relative cell number",title="Abstracted Data from the paper")


df_3 = CSV.read("QS_flow.csv",DataFrame,header = false, delim = ";",decimal=',',types = [Float64,Float64])
m_3 = Matrix(df_3)
x_3 = m_3[:,1]
y_3 = m_3[:,2]
scatter(x_3,y_3,color=:blue,label = "0.5 Flow",xaxis="Time (h)",yaxis="Normalized QS output",title="Abstracted Data from the paper")

df_4 = CSV.read("QS_no_flow.csv",DataFrame,header = false, delim = ";",decimal=',',types = [Float64,Float64])
m_4 = Matrix(df_4)
x_4 = m_4[:,1]
y_4 = m_4[:,2]
scatter!(x_4,y_4,color=:red,label = "No Flow",xaxis="Time (h)",yaxis="Normalized QS output",title="Abstracted Data from the paper")



"""
Simplified ODE Model for the Data Fitting 
"""
function sqs_ft_fit(du,u,p,t)
    AIP_f,N_f = u
    K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, r, K_f, κ,δ, η = p
    du[1] = (α_D+β_D*κ*AIP_f^2/(K_AgrAP+κ*AIP_f^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f*(α_C+β_C*AIP_f^2/(K_AgrAP+AIP_f^2))/(γ_C+K_C*AIP_f))*N_f - γ_AIP*AIP_f - η*AIP_f
    du[2] = (r+δ)*N_f*(1-N_f/K_f)
end

"""
Comparing the output of the ODE Model with no flow to the Data with the mean squarred error
"""
function residuals(p)
    df = CSV.read("QS_no_flow.csv",DataFrame,header = false, delim = ";",decimal=',',types = [Float64,Float64])
    m = Matrix(df)

    tspan = (0.0,8.0)
    init = [0.3,1.0]
    parms = [p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],0,0]
    sqs_ft_prob = ODEProblem(sqs_ft_fit,init,tspan,parms)
    sqs_ft_sol = solve(sqs_ft_prob,Rosenbrock23(),tstops = m[:,1],saveat = m[:,1],verbose = true,dt = 0.00001,maxiters = 5e5)

    model_values = sqs_ft_sol[1,:]
    p = scatter(sqs_ft_sol.t,model_values,label="Fitted Points", color =:green)
    x = m[:,1]
    y = m[:,2]
    scatter!(x,y,xaxis="t",yaxis="QS output",label="Abstracted data from the paper", color=:red)
    display(p)
    return mse(model_values, m[:, 2])
end

# Data Fitting 
lower = zeros(14)
upper = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]
result = optimize(residuals,[0.13450556291482174, 0.8423345442637511, 0.17389870269910346, 1.2885102737641, 0.8509806875395725, 0.0573546243722467, 3.9360769650398657, 2.246255392814358, 0.5589779627505864, 2.3224079996776235, 1.541090044508264, 1.091454061706784, 4.027995074988411, 1.])
o = Optim.minimizer(result)
residuals([o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14]])


"""
Comparing the output of the ODE Model with flow to the Data with the mean squarred error
"""
function residuals2(p)
    df = CSV.read("QS_flow.csv",DataFrame,header = false, delim = ";",decimal=',',types = [Float64,Float64])
    m = Matrix(df)

    tspan = (0.0,8.0)
    init = [0.2,1.0]
    parms = [o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14],p[1],p[2]]
    sqs_ft_prob = ODEProblem(sqs_ft_fit,init,tspan,parms)
    sqs_ft_sol = solve(sqs_ft_prob,Rosenbrock23(),tstops = m[:,1],saveat = m[:,1],verbose = true,dt = 0.00001,maxiters = 5e7)

    model_values = sqs_ft_sol[1,:]
    p = scatter(sqs_ft_sol.t,model_values,label="Fitted Points", color =:green)
    x = m[:,1]
    y = m[:,2]
    scatter!(x,y,xaxis="t",yaxis="QS output",label="Abstracted data from the paper", color=:blue)
    display(p)
    return mse(model_values, m[:, 2])
end

# Data Fitting
lower2 = zeros(2)
upper2 = [Inf,Inf]
result2 = optimize(residuals2,[23330.836191188435, 0.28056625768767596])
o2 = Optim.minimizer(result2)
residuals2(o2)



# -----------------------------------------------------------------------------------
# Bifurcation Analysis 

# Bifurcation for the one dimensional model
function solo_AIP(x,p) 
    @unpack p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12 = p
    x = p6.+p7.*p12.*x.^2 ./(p5.+p12.*x.^2).-p8.*(-K_1(p7,p6,p8,p10,p11)/(λ(p7,p6,p8,p10,p11,p5).*p12.*x.^2 .+K_2(p8)).+K_3(p7,p6,p8,p10,p11)).-p9.*x .- p1.*x.*(p2.+p3*x.^2 ./(p5.+x.^2))./(p4.+p1.*x)
    return x
end
par_solo_AIP = (p1 = 2., p2 = 1.8, p3 = 0.01, p4 = 2., p5 = 2., p6 = 0.8, p7 = 5.5, p8 = 0.1, p9 = 1., p10 = 2., p11 = 3.,p12=1.)
bifur_prob = BifurcationKit.BifurcationProblem(solo_AIP, [2.5], par_solo_AIP,(@lens _.p1), record_from_solution = (x,p) -> (x = x[1]))
opt_newton = NewtonPar(tol = 1e-10, max_iterations = 10000)
opts = ContinuationPar(p_min = 0., p_max = 15.,dsmax=.2,ds=.2,dsmin=0.001)
br = BifurcationKit.continuation(bifur_prob, PALC(), opts,max_steps = 1000,bothsides = true)
plot(br,xaxis = L"K_C",yaxis = L"x^*",legend=false)


# Bifurcation for the multi-dimensional model
function sqs_ft!(du,u,p,t = 0)
    AIP_f,AIP_C1,AIP_C2,N_f,N_C1,N_C2 = u
    @unpack K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, r, K_f,K_C1,K_C2, δ, μ, ν, η, σ, ρ, κ = p
    du[1] = (α_D+β_D*κ*AIP_f^2/(K_AgrAP+κ*AIP_f^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f*(α_C+β_C*AIP_f^2/(K_AgrAP+AIP_f^2))/(γ_C+K_C*AIP_f))*N_f - γ_AIP*AIP_f - η*AIP_f - σ*AIP_f + ρ*AIP_C1 - σ*AIP_f + ρ*AIP_C2
    du[2] = (α_D+β_D*κ*AIP_C1^2/(K_AgrAP+κ*AIP_C1^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C1*(α_C+β_C*AIP_C1^2/(K_AgrAP+AIP_C1^2))/(γ_C+K_C*AIP_C1))*N_C1 -γ_AIP*AIP_C1 + σ*AIP_f - ρ*AIP_C1
    du[3] = (α_D+β_D*κ*AIP_C2^2/(K_AgrAP+κ*AIP_C2^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C2*(α_C+β_C*AIP_C2^2/(K_AgrAP+AIP_C2^2))/(γ_C+K_C*AIP_C2))*N_C2 -γ_AIP*AIP_C2 + σ*AIP_f - ρ*AIP_C2
    du[4] = (r+δ)*N_f*(1-N_f/K_f)-μ*N_f+ν*N_C1-μ*N_f+ν*N_C2
    du[5] = r*N_C2*(1-N_C1/K_C1)+μ*N_f-ν*N_C1
    du[6] = r*N_C2*(1-N_C2/K_C2)+μ*N_f-ν*N_C2
    du
end

init = [1.0,1.0,1.0,1.,1.,1.]

p = round.(rand(22),digits=2)
parms_bifurcation = (
    K_C = p[1], α_C = p[2], β_C = p[3], γ_C = p[4], K_AgrAP = p[5], α_D = p[6], β_D = p[7], γ_D = p[8], γ_AIP = p[9],
    K_m = p[10], vmax = p[11], r = p[12], K_f = p[13], K_C1 = p[14], K_C2 = p[15], δ = p[16], μ = p[17], ν = p[18], η = .1, σ = p[20], ρ = p[21], κ = p[22]
)

parms_bifurcation = (
    K_C = .1, α_C = 1.8, β_C = 0.01, γ_C = 2.0, K_AgrAP = 2.0, α_D = 0.8, β_D = 5.5, γ_D = 0.1, γ_AIP = 1.0,
    K_m = 2.0, vmax = 3.0, r = 1.0, K_f = 1.0, K_C1 = 1.0, K_C2 = 1.0, δ = 1.0, μ = 1.0, ν = 1.0, η = .1, σ = 1.0, ρ = 1.0, κ = 1.
)

parms_bifurcation = (K_C = 0.47, α_C = 0.87, β_C = 0.34, γ_C = 0.4, K_AgrAP = 0.74, α_D = 0.39, β_D = 0.42, γ_D = 0.71, γ_AIP = 0.47, K_m = 0.22, vmax = 0.11, r = 0.08, K_f = 0.45, K_C1 = 0.01, K_C2 = 0.84, δ = 0.64, μ = 0.61, ν = 0.76, η = 0.1, σ = 0.72, ρ = 0.73, κ = 1.)

# Bifurcation problem for x[1]
sqs_ft_bifur_prob_1 = BifurcationKit.BifurcationProblem(
    sqs_ft!, init, parms_bifurcation, (@lens _.K_C), 
    record_from_solution = (x,p) -> (x = x[1])
)

# Bifurcation problem for x[2]
sqs_ft_bifur_prob_2 = BifurcationKit.BifurcationProblem(
    sqs_ft!, init, parms_bifurcation, (@lens _.K_C), 
    record_from_solution = (x,p) -> (x = x[2])
)

# Bifurcation problem for x[2]
sqs_ft_bifur_prob_3 = BifurcationKit.BifurcationProblem(
    sqs_ft!, init, parms_bifurcation, (@lens _.K_C), 
    record_from_solution = (x,p) -> (x = x[3])
)

# Options for bifurcation continuation
opts_br = ContinuationPar(
    p_min = 0.0, p_max = 100.0, max_steps = 5000, dsmin = 0.0001, dsmax = 0.01, ds = 0.0005
)

# Bifurcation diagrams
diagram_1 = bifurcationdiagram(sqs_ft_bifur_prob_1, PALC(),6, (args...) -> setproperties(opts_br))
diagram_2 = bifurcationdiagram(sqs_ft_bifur_prob_2, PALC(),6, (args...) -> setproperties(opts_br))
diagram_3 = bifurcationdiagram(sqs_ft_bifur_prob_3, PALC(),6, (args...) -> setproperties(opts_br))


# Combine the two diagrams into a single plot
combined_diagram = plot(diagram_1; code = (), legend = true,label = L"AIP_f",xaxis=L"K_{C}")
plot!(diagram_2; code = (), legend = true,label = L"AIP_{C_1}",xaxis=L"K_{C}")
plot!(diagram_3; code = (), legend = true,label = L"AIP_{C_2}",xaxis=L"K_{C}",yaxis=L"x^*")



# -----------------------------------------------------------------------------------
# Delay Differential Equation


# 1D DDE
function solo_AIP_delay(du,u,h,p,t)
    AIP = u
    K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, κ, τ = p
    hist1 = h(p, t - τ)[1]
    du .= (α_D.+β_D.*κ.*hist1.^2 ./(K_AgrAP.+κ.*hist1.^2).-γ_D.*(-K_1(β_D,α_D,γ_D,K_m,vmax) ./(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP).*κ.*AIP.^2 .+K_2(γ_D)) .+K_3(β_D,α_D,γ_D,K_m,vmax)).-K_C.*AIP.*(α_C.+β_C.*AIP.^2 ./(K_AgrAP.+AIP.^2))./(γ_C.+K_C.*AIP)).-γ_AIP.*AIP
end
tspan = (0.0,20.0)
init = [1]
parms = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.,1.,1.,1.,2.]
h(p,t) = 1 .*ones(1,1000)
τ = 2
lags = [τ]
solo_AIP_delay_prob = DDEProblem(solo_AIP_delay,init,h,tspan,parms,constant_lags=lags)
alg = MethodOfSteps(Tsit5())
solo_AIP_delay_sol = DifferentialEquations.solve(solo_AIP_delay_prob,alg)
p = plot(solo_AIP_delay_sol,label = L"AIP",linewidth = 2,dpi = 300)


# Multi-dimensional DDE
function sqs_ft_delay(du,u,h,p,t)
    AIP_f,AIP_C1,AIP_C2,N_f,N_C1,N_C2 = u
    K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, r, K_f, K_C1, K_C2, δ, μ, ν, η, σ, ρ,κ, τ = p
    hist1 = h(p, t - τ)[1]
    hist2 = h(p, t - τ)[2]
    hist3 = h(p, t - τ)[3]
    du[1] = (α_D+β_D*κ*hist1^2/(K_AgrAP+κ*hist1^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f*(α_C+β_C*AIP_f^2/(K_AgrAP+AIP_f^2))/(γ_C+K_C*AIP_f))*N_f - γ_AIP*AIP_f - η*AIP_f - σ*AIP_f + ρ*AIP_C1 - σ*AIP_f + ρ*AIP_C2
    du[2] = (α_D+β_D*κ*hist2^2/(K_AgrAP+κ*hist2^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C1*(α_C+β_C*AIP_C1^2/(K_AgrAP+AIP_C1^2))/(γ_C+K_C*AIP_C1))*N_C1 -γ_AIP*AIP_C1 + σ*AIP_f - ρ*AIP_C1
    du[3] = (α_D+β_D*κ*hist3^2/(K_AgrAP+κ*hist3^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C2*(α_C+β_C*AIP_C2^2/(K_AgrAP+AIP_C2^2))/(γ_C+K_C*AIP_C2))*N_C2 -γ_AIP*AIP_C2 + σ*AIP_f - ρ*AIP_C2
    du[4] = (r+δ)*N_f*(1-N_f/K_f)-μ*N_f+ν*N_C1-μ*N_f+ν*N_C2
    du[5] = r*N_C2*(1-N_C1/K_C1)+μ*N_f-ν*N_C1
    du[6] = r*N_C2*(1-N_C2/K_C2)+μ*N_f-ν*N_C2
end
tspan = (0.0,5000.0)
init = [1,1,1,0.1,0.1,0.1]
parms = [.1,1.,1.,1.,1.,1.,10.,0.77,0.01,1.,1.,1.,.01,1.,.01,.1,.01,.01,.1,.01,.01,1.,300]
h(p,t) = ones(6,1000)
τ = 300
lags = [τ]
sqs_ft_prob_delay = DDEProblem(sqs_ft_delay,init,h,tspan,parms,constant_lags=lags)
alg = MethodOfSteps(Tsit5())
sqs_ft_sol_delay = DifferentialEquations.solve(sqs_ft_prob_delay,alg)
p = plot(sqs_ft_sol_delay,label = [L"AIP_f" L"AIP_{C_1}" L"AIP_{C_2}" L"N_f" L"N_{C_1}" L"N_{C_2}"],linewidth = 2,legend = :outertopright,dpi = 300)


# Bifurcation of the DDE
function bifur_sqs_ft_delay(u,xd,p)
    AIP_f,AIP_C1,AIP_C2,N_f,N_C1,N_C2 = u
    @unpack K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, r, K_f, K_C1, K_C2, δ, μ, ν, η, σ, ρ,κ, τ = p
    [
        (α_D+β_D*κ*xd[1][1]^2/(K_AgrAP+κ*xd[1][1]^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f*(α_C+β_C*AIP_f^2/(K_AgrAP+AIP_f^2))/(γ_C+K_C*AIP_f))*N_f - γ_AIP*AIP_f - η*AIP_f - σ*AIP_f + ρ*AIP_C1 - σ*AIP_f + ρ*AIP_C2,
        (α_D+β_D*κ*xd[1][2]^2/(K_AgrAP+κ*xd[1][2]^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C1*(α_C+β_C*AIP_C1^2/(K_AgrAP+AIP_C1^2))/(γ_C+K_C*AIP_C1))*N_C1-γ_AIP*AIP_C1+σ*AIP_f-ρ*AIP_C1,
        (α_D+β_D*κ*xd[1][3]^2/(K_AgrAP+κ*xd[1][3]^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C2*(α_C+β_C*AIP_C2^2/(K_AgrAP+AIP_C2^2))/(γ_C+K_C*AIP_C2))*N_C2-γ_AIP*AIP_C2+σ*AIP_f-ρ*AIP_C2,
        (r+δ)*N_f*(1-N_f/K_f)-μ*N_f+ν*N_C1-μ*N_f+ν*N_C2,
        r*N_C2*(1-N_C1/K_C1)+μ*N_f-ν*N_C1,
        r*N_C2*(1-N_C2/K_C2)+μ*N_f-ν*N_C2
    ]
end
delays(par) = [par.τ]
parms_bifurcation_delay = (K_C=.1,α_C=1.,β_C=1.,γ_C=1.,K_AgrAP=1.,α_D=1.,β_D=10.,γ_D=0.77,γ_AIP=0.01,K_m=1.,vmax=1.,r=1.,K_f=.01,K_C1=1.,K_C2=1.,δ=.1,μ=.01,ν=.01,η=.1,σ=.01,ρ=.01,κ=1., τ=2.)
init = [1,1,1,0.1,0.1,0.1]
sqs_ft_bifur_prob_delay = ConstantDDEBifProblem(bifur_sqs_ft_delay,delays,init,parms_bifurcation_delay,(@lens _.τ),record_from_solution = (x,p) -> (x = x[1]))
optn = NewtonPar(eigsolver = DDE_DefaultEig(maxit = 1000))
opts = DDEBifurcationKit.ContinuationPar(p_max = 100., p_min = 0., newton_options = optn,ds = 0.01, dsmax = 0.2,dsmin=0.01)
br = DDEBifurcationKit.continuation(sqs_ft_bifur_prob_delay, PALC(), opts; plot = false)
scene = plot(br)


# DDE Bifurcation Condition
function DDE_Bifur_Cond(K_C,α_C,β_C,γ_C,K_AgrAP,α_D,β_D,γ_D,γ_AIP,K_m,vmax,r,K_f,K_C1,K_C2,δ,μ,ν,η,σ,ρ,κ)

    function qs_ft(x)
        [(p[6]+p[7]*p[22]*x[1]^2/(p[5]+p[22]*x[1]^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x[1]^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x[1]*(p[2]+p[3]*x[1]^2/(p[5]+x[1]^2))/(p[4]+p[1]*x[1]))*x[4]-p[9]*x[1]-p[19]*x[1]-p[20]*x[1]+p[21]*x[2]-p[20]*x[1]+p[21]*x[3],
        (p[6]+p[7]*p[22]*x[2]^2/(p[5]+p[22]*x[2]^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x[2]^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x[2]*(p[2]+p[3]*x[2]^2/(p[5]+x[2]^2))/(p[4]+p[1]*x[2]))*x[5]-p[9]*x[2]+p[20]*x[1]-p[21]*x[2],
        (p[6]+p[7]*p[22]*x[3]^2/(p[5]+p[22]*x[3]^2)-p[8]*(-K_1(p[7],p[6],p[8],p[10],p[11])/(λ(p[7],p[6],p[8],p[10],p[11],p[5])*p[22]*x[3]^2+K_2(p[8]))+K_3(p[7],p[6],p[8],p[10],p[11]))-p[1]*x[3]*(p[2]+p[3]*x[3]^2/(p[5]+x[3]^2))/(p[4]+p[1]*x[3]))*x[6]-p[9]*x[3]+p[20]*x[1]-p[21]*x[3],
        (p[12]+p[16])*x[4]*(1-x[4]/p[13])-p[17]*x[4]+p[18]*x[5]-p[17]*x[4]+p[18]*x[6],
        p[12]*x[6]*(1-x[5]/p[14])+p[17]*x[4]-p[18]*x[5],
        p[12]*x[6]*(1-x[6]/p[15])+p[17]*x[4]-p[18]*x[6]]
    end
    p = [K_C,α_C,β_C,γ_C,K_AgrAP,α_D,β_D,γ_D,γ_AIP,K_m,vmax,r,K_f,K_C1,K_C2,δ,μ,ν,η,σ,ρ,κ]
    sol = nlsolve(qs_ft, [10., 10., 10., 10., 10., 10.])
    z = sol.zero

    s(x) = (-(2*K_1(β_D,α_D,γ_D,K_m,vmax)*γ_D*λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*x[1])/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)+x[1]^2+K_2(γ_D))^2
            -K_C*((β_C+α_C)*γ_C*κ^2*x[1]^4+2*K_AgrAP*K_C*β_C*κ*x[1]^3+(3*K_AgrAP*β_C+2*K_AgrAP*α_C)*γ_C*κ*x[1]^2+2*K_AgrAP*α_C*γ_C)/((K_C*x[1]+γ_C)^2*(κ*x[1]^2+K_AgrAP)^2)
            )*x[2]-γ_AIP-η-2*σ

    d(x) = x[2]*2*K_AgrAP*β_D*κ*x[1]/(κ*x[1]^2+K_AgrAP)^2

    a = s([z[1],z[4]])
    b = s([z[2],z[5]])
    c = s([z[3],z[6]])
    d1 = d([z[1],z[4]])
    d2 = d([z[2],z[5]])
    d3 = d([z[3],z[6]])

    P(λ) = -a*b*c-a*b*λ-a*c*λ-b*c*λ-b*λ^2-c*λ^2+a*λ^2+b*ρ*σ+c*ρ*σ+λ^3-2*λ*ρ*σ
    Q_1(λ) = -a*b*d3-a*c*d2-a*d2*λ-a*d3*λ-b*c*d1-b*d1*λ-b*d3*λ-c*d1*λ-c*d1*λ-c*d2*λ-d1*λ^2-d2*λ^2-d3*λ^2+d2*ρ*σ+d3*ρ*σ
    Q_2(λ) = -a*d2*d3-b*d1*d3-c*d1*d2+d1*d2*λ+d1*d3*λ+d2*d3*λ
    Q_3(λ) = -d1*d2*d3

    y = range(0, 5, length=10000)im
    z_P = abs2.(P.(y))
    z_Q1 = abs2.(Q_1.(y))
    z_Q2 = abs2.(Q_2.(y))
    z_Q3 = abs2.(Q_3.(y))
    pl = plot(range(0,5, length=10000),z_P,label=L"|P(iy)|^2",yaxis=:log,xaxis="y")
    plot!(range(0, 5, length=10000),3 .*(z_Q1+z_Q2+z_Q3),label=L"3(|Q_1(iy)|^2+|Q_2(iy)|^2+|Q_3(iy)|^2)",legend=:outerbottom,yaxis=:log)
    display(pl)

    if all(z_P .< 3 .*(z_Q1+z_Q2+z_Q3))
        display(true)
    end
end

DDE_Bifur_Cond(0.1,1.,1.,1.,1.,1.,10.,0.77,0.01,1.,1.,1.,.01,1.,.01,.1,.01,.01,.1,.01,.01,1.)



# Second DDE Model
function sqs_ft_delay_f(du,u,h,p,t)
    AIP_f1,AIP_f2,AIP_C1,AIP_C2,N_f1,N_f2,N_C1,N_C2 = u
    K_C, α_C, β_C, γ_C, K_AgrAP, α_D, β_D, γ_D, γ_AIP, K_m, vmax, r, K_f, K_C1, K_C2, δ, μ, ν, η, σ, ρ,ξ,κ, τ_1,τ_2 = p
    hist1 = h(p, t - τ_1)[1]    # AIP_f1(t-τ_1)
    hist2 = h(p, t - τ_2)[2]    # AIP_f1(t-τ_2)
    hist3 = h(p, t - τ_2)[3]    # AIP_f2(t-τ_2)
    hist4 = h(p, t - τ_2)[4]    # AIP_C1(t-τ_2)
    hist5 = h(p, t - τ_2)[5]    # AIP_C2(t-τ_2)
    du[1] = (α_D+β_D*κ*AIP_f1^2/(K_AgrAP+κ*AIP_f1^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f1*(α_C+β_C*AIP_f1^2/(K_AgrAP+AIP_f1^2))/(γ_C+K_C*AIP_f1))*N_f1 - γ_AIP*AIP_f1 -ξ*AIP_f1 - σ*AIP_f1 + ρ*e^(-γ_AIP*τ_2)*hist4
    du[2] = (α_D+β_D*κ*AIP_f2^2/(K_AgrAP+κ*AIP_f2^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_f2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_f2*(α_C+β_C*AIP_f2^2/(K_AgrAP+AIP_f2^2))/(γ_C+K_C*AIP_f2))*N_f2 - γ_AIP*AIP_f2 - η*AIP_f2 +ξ*e^(-γ_AIP*τ_1)*hist1 - σ*AIP_f2 + ρ*e^(-γ_AIP*τ_2)*hist5
    du[3] = (α_D+β_D*κ*AIP_C1^2/(K_AgrAP+κ*AIP_C1^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C1^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C1*(α_C+β_C*AIP_C1^2/(K_AgrAP+AIP_C1^2))/(γ_C+K_C*AIP_C1))*N_C1 -γ_AIP*AIP_C1 + σ*e^(-γ_AIP*τ_2)*hist2 - ρ*AIP_C1
    du[4] = (α_D+β_D*κ*AIP_C2^2/(K_AgrAP+κ*AIP_C2^2)-γ_D*(-K_1(β_D,α_D,γ_D,K_m,vmax)/(λ(β_D,α_D,γ_D,K_m,vmax,K_AgrAP)*κ*AIP_C2^2+K_2(γ_D))+K_3(β_D,α_D,γ_D,K_m,vmax))-K_C*AIP_C2*(α_C+β_C*AIP_C2^2/(K_AgrAP+AIP_C2^2))/(γ_C+K_C*AIP_C2))*N_C2 -γ_AIP*AIP_C2 + σ*e^(-γ_AIP*τ_2)*hist3 - ρ*AIP_C2
    du[5] = (r+δ)*N_f1*(1-N_f1/K_f)-μ*N_f1+ν*N_C1
    du[6] = r*N_f2*(1-N_f2/K_f)-μ*N_f2+ν*N_C2
    du[7] = r*N_C1*(1-N_C1/K_C1)+μ*N_f1-ν*N_C1
    du[8] = r*N_C2*(1-N_C2/K_C2)+μ*N_f2-ν*N_C2
end
tspan = (0.0,20.0)
init = [0,0,0,0,0.1,0.1,0.1,0.1]
parms = push!(round.(rand(22),digits=2),1,2)
parms = [0.79, 0.8, 0.1, 0.89, 0.79, 0.9, 0.04, 0.06, 0.08, 0.63, 0.6, 0.69, 0.48, 0.22, 1.0, 0.78, 0.57, 0.32, 0.99, 0.81, 0.72, 0.47,1., 1.0, 2.0]
# parms = [0.87, 0.03, 0.57, 0.38, 0.98, 0.49, 0.27, 0.68, 0.08, 0.97, 0.9, 0.46, 0.79, 0.01, 0.74, 0.37, 0.08, 0.96, 0.78, 0.74, 0.44, 0.37,1., 10.0, 2.0]
h(p,t) = ones(5,1000)
τ_1 = 1
τ_2 = 2
lags = [τ_1,τ_2]
sqs_ft_prob_delay = DDEProblem(sqs_ft_delay_f,init,h,tspan,parms,constant_lags=lags)
alg = MethodOfSteps(Rosenbrock23())
sqs_ft_sol_delay = DifferentialEquations.solve(sqs_ft_prob_delay,alg)
p = plot(sqs_ft_sol_delay,label = [L"AIP_{f_1}" L"AIP_{f_2}" L"AIP_{C_1}" L"AIP_{C_2}" L"N_{f_1}" L"N_{f_2}" L"N_{C_1}" L"N_{C_2}"],linewidth = 2,legend = :outertopright,dpi = 300)




# Visualization of different Hill Functions
# Define the Hill function
function hill_function(x, Vmax, Kd, n)
    return (Vmax * x^n) / (Kd^n + x^n)
end
x = range(0, stop=10, length=500)
parameter_sets = [
    (1.0, 2.0, 1.0), # Vmax=1, Kd=2, n=1
    (1.0, 2.0, 2.0), # Vmax=1, Kd=2, n=2
    (1.0, 2.0, 4.0), # Vmax=1, Kd=2, n=4
    (1.0, 4.0, 2.0), # Vmax=1, Kd=4, n=2
    (2.0, 2.0, 2.0)  # Vmax=2, Kd=2, n=2
]
plot(x, hill_function.(x, parameter_sets[1]...), label="L=1.0, K=2.0, n=1.0",legend=:outertopright)
for (Vmax, Kd, n) in parameter_sets[2:end]
    plot!(x, hill_function.(x, Vmax, Kd, n), label="L=$Vmax, K=$Kd, n=$n")
end
title!("Hill Function Variants")
xaxis!("x")
yaxis!("H(x)")
display(plot)



end 