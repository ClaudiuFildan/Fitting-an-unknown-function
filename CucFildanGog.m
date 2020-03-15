%% Identification

% Closes and clears all the previously loaded data
clear all;
close all;

% Loads the identification and validation data
load('proj_fit_09.mat');

% Creates 2 vectors which hold the input coordinates, corresponding to the X
% and Y axes, the dimension of the input data and the output matrix of the
% non-linear function
Xone = id.X{1,1};
Xtwo = id.X{2,1};
Xdim = id.dims(1);
Ymatrix = id.Y;

% Plots the three-dimensional representation of the data
surf(Xone, Xtwo, Ymatrix);
title('Initial Identification Data'); xlabel('Input X_1');ylabel('Input X_2');zlabel('Output Y');

% We select the dimension of the matrix holding the center points
Cdim = 7;

% Creates a vector of zeros 
% Selects the evenly spaced points forming the centers of the RBFs 
Cone = zeros(1,Cdim); 
Cone = linspace(Xone(1),Xone(end),Cdim);

% Computes the distance between 2 centers of the RBFs
b1 = abs(Cone(2)-Cone(1));
b2 = b1;

% Cflat creates a 2 by N^2 matrix having all the combinations of centers
% Xflat creates a 2 by n^2 matrix having all the combinations of input data
% Yflat creates a 1 by n^2 vector having all the combination of output data 
Cflat = zeros(2,Cdim^2);
Xflat = zeros(2,Xdim^2);
Yflat = zeros(1,Xdim^2);

% Creates a matrix which on its first row has the first term of the centers by n times followed by the next ones,
% also taken by n times. On the second row the terms are placed in their initial order, rewritten by
% n times. This way, we have access to all the combinations of the center points
for k = 1:1:Cdim
    for j = 1:1:Cdim
       Cflat(1,(k-1)*Cdim+j) = Cone(k);
       Cflat(2,(k-1)*Cdim+j) = Cone(j);
    end
end

% Creates a matrix which on its first row has the first term of the input coordinates by n times followed by the next ones,
% also taken by n times. On the second row the terms are placed in their initial order, rewritten by
% n times. This way, we have access to all the combinations of the input data points   
for k = 1:1:Xdim
    for j = 1:1:Xdim
        Xflat(1,(k-1)*Xdim+j) = Xone(k);
        Xflat(2,(k-1)*Xdim+j) = Xtwo(j);
    end
end 
 
% Creates a vector which contains all the output data from the initial marix, concatenating all the rows
% of the initial matrix one after the other, making it easier to access the outputs
for k = 1:1:Xdim
    for j = 1:1:Xdim
       Yflat((k-1)*Xdim+j) = Ymatrix(k,j); 
    end    
end    
     
% Creates a n^2 x N^2 matrix having all the RBFs corresponding to each center on the grid.
% Taking the matrix as having element which have the following notations Phi_k(x_i) 
% Two counters are taken, the first one going from 1 to n^2, being the index of the rows representing the RBFs(regressors)
% and a second one which is going from 1 to N^2 being the index of the
% colums representing the position of the elements from the data coordinates
Phi = zeros(Xdim^2,Cdim^2);
for i = 1:1:Xdim^2
    for k = 1:1:Cdim^2
        Phi(i,k) = exp(-((Cflat(1,k)-Xflat(1,i))/b1)^2-((Cflat(2,k)-Xflat(2,i))/b2)^2);           
    end    
end    

% A parameter vector with an unknown value is computed by applying linear
% regression methods with the RBF matrix and the transposed matrix of the
% output. We also compute the approximated model, Yhat, of the output checking the
% correctness of the previously applied linear regression methods by
% multiplying the regressors with the parameters
Theta = Phi\Yflat';
Yhat = Phi*Theta;

% Because we transformed our matrices in vectors previously, our output
% approximation is also a vector, but in order to display the values it
% must be converted back to a matrix form
YhatMatrix = zeros(Xdim,Xdim);
for i = 1:1:Xdim
   for j = 1:1:Xdim
       YhatMatrix(i,j) = Yhat((i-1)*Xdim+j);
   end    
end

% Plots the approximation of the identification model
figure
surf(Xone,Xtwo, YhatMatrix);
title('Approximated Identification Data'); xlabel('Input X_1');ylabel('Input X_2');zlabel('Output Y');

% Calculates the difference between the initial values of the function and
% our approximated values to see if the approximated model resembles the
% initial one, being aware of the difference caused by noise
Mse = sum((Yflat-Yhat').^2)/length(Yflat);

%% Validation

% We follow the same procedure as in the identification part in obtaining
% the regressor matrix
XoneVal = val.X{1,1};
XtwoVal = val.X{2,1};
XdimVal = val.dims(1);
YmatrixVal = val.Y;

figure
surf(XoneVal, XtwoVal, YmatrixVal);
title('Initial Validation Data'); xlabel('Input X_1');ylabel('Input X_2');zlabel('Output Y');


% We keep the same dimension for the matrix of centers
CdimVal = Cdim;

ConeVal = zeros(1,CdimVal); 
ConeVal = linspace(XoneVal(1),XoneVal(end),CdimVal);

b1Val = abs(ConeVal(2)-ConeVal(1));
b2Val = b1Val;

CflatVal = zeros(2,CdimVal^2);
XflatVal = zeros(2,XdimVal^2);
YflatVal = zeros(1,XdimVal^2);
  
for k = 1:1:CdimVal
    for j = 1:1:CdimVal
       CflatVal(1,(k-1)*CdimVal+j) = ConeVal(k);
       CflatVal(2,(k-1)*CdimVal+j) = ConeVal(j);
    end
end

for k = 1:1:XdimVal
    for j = 1:1:XdimVal
        XflatVal(1,(k-1)*XdimVal+j) = XoneVal(k);
        XflatVal(2,(k-1)*XdimVal+j) = XtwoVal(j);
    end
end 

for k = 1:1:XdimVal
    for j = 1:1:XdimVal
       YflatVal((k-1)*XdimVal+j) = YmatrixVal(k,j); 
    end    
end    
        
PhiVal = zeros(XdimVal^2,CdimVal^2);
for i = 1:1:XdimVal^2
    for k = 1:1:CdimVal^2
        PhiVal(i,k) = exp(-((CflatVal(1,k)-XflatVal(1,i))/b1)^2-((CflatVal(2,k)-XflatVal(2,i))/b2)^2);           
    end    
end    

% Using the Theta from identification part and the regressor for the new
% set of data, we calculate the approximation of the validation output data
YhatVal = PhiVal*Theta;

YhatMatrixVal = zeros(XdimVal,XdimVal);
for i = 1:1:XdimVal
   for j = 1:1:XdimVal
       YhatMatrixVal(i,j) = YhatVal((i-1)*XdimVal+j);
   end    
end

figure
surf(XoneVal,XtwoVal, YhatMatrixVal);
title('Approximated Validation Data'); xlabel('Input X_1');ylabel('Input X_2');zlabel('Output Y');

% Calculates the difference between the initial validation data of the function and
% our approximated values using the computed values in identification for theta
MseVal = sum((YflatVal-YhatVal').^2)/length(YflatVal);

%% Optimal Mean Square error

% CdimFrom is the minimal dimension of the centers matrix from which we compute the optimal MSE
% CdimTo is the value of the maximal dimension of the centers matrix
% MseV is the vector in which we place all the values of the identification error
% MseValV is the vector in which we place all the values of the validation error

CdimFrom = 2;
CdimTo = Xdim-1;
MseV = zeros(1,(CdimTo-CdimFrom));
MseValV = zeros(1,(CdimTo-CdimFrom));
 
% We use the previously presented procedure for identification and
% validation, checking the optimal MSE for a series of different number of
% centers 
for CdimOp=CdimFrom:1:CdimTo
        
        Xone = id.X{1,1};
        Xtwo = id.X{2,1};
        Xdim = id.dims(1);
        Ymatrix = id.Y;
        
        Cdim = CdimOp;
        Cone = zeros(1,Cdim); 
        Cone = linspace(Xone(1),Xone(end),Cdim);

        b1 = abs(Cone(2)-Cone(1));
        b2 = b1;

        Cflat = zeros(2,Cdim^2);
        Xflat = zeros(2,Xdim^2);
        Yflat = zeros(1,Xdim^2);

        for k = 1:1:Cdim
            for j = 1:1:Cdim
                Cflat(1,(k-1)*Cdim+j) = Cone(k);
                Cflat(2,(k-1)*Cdim+j) = Cone(j);
            end
        end
 
        for k = 1:1:Xdim
            for j = 1:1:Xdim
                Xflat(1,(k-1)*Xdim+j) = Xone(k);
                Xflat(2,(k-1)*Xdim+j) = Xtwo(j);
            end
        end 
 
        for k = 1:1:Xdim
            for j = 1:1:Xdim
                Yflat((k-1)*Xdim+j) = Ymatrix(k,j); 
            end    
        end    
         
        Phi = zeros(Xdim^2,Cdim^2);
        
        for i = 1:1:Xdim^2
            for k = 1:1:Cdim^2
                Phi(i,k) = exp(-((Cflat(1,k)-Xflat(1,i))/b1)^2-((Cflat(2,k)-Xflat(2,i))/b2)^2);           
            end    
        end    

        Theta = Phi\Yflat';
        Yhat = Phi*Theta;

        YhatMatrix = zeros(Xdim,Xdim);
        for i = 1:1:Xdim
            for j = 1:1:Xdim
                YhatMatrix(i,j) = Yhat((i-1)*Xdim+j);
            end    
        end

        Mse = sum((Yflat-Yhat').^2)/length(Yflat);
        MseV(Cdim) = Mse;
    
        XoneVal = val.X{1,1};
        XtwoVal = val.X{2,1};
        XdimVal = val.dims(1);
        YmatrixVal = val.Y;

        CdimVal = CdimOp;

        ConeVal = zeros(1,CdimVal); 
        ConeVal = linspace(XoneVal(1),XoneVal(end),CdimVal);

        b1Val = abs(ConeVal(2)-ConeVal(1));
        b2Val = b1Val;

        CflatVal = zeros(2,CdimVal^2);
        XflatVal = zeros(2,XdimVal^2);
        YflatVal = zeros(1,XdimVal^2);
 
        for k = 1:1:CdimVal
            for j = 1:1:CdimVal
                CflatVal(1,(k-1)*CdimVal+j) = ConeVal(k);
                CflatVal(2,(k-1)*CdimVal+j) = ConeVal(j);
            end
        end
   
        for k = 1:1:XdimVal
            for j = 1:1:XdimVal
                XflatVal(1,(k-1)*XdimVal+j) = XoneVal(k);
                XflatVal(2,(k-1)*XdimVal+j) = XtwoVal(j);
            end
        end 
 
        for k = 1:1:XdimVal
            for j = 1:1:XdimVal
                YflatVal((k-1)*XdimVal+j) = YmatrixVal(k,j); 
            end    
        end    
         
        PhiVal = zeros(XdimVal^2,CdimVal^2);
        
        for i = 1:1:XdimVal^2
            for k = 1:1:CdimVal^2
                PhiVal(i,k) = exp(-((CflatVal(1,k)-XflatVal(1,i))/b1)^2-((CflatVal(2,k)-XflatVal(2,i))/b2)^2);           
            end    
        end    

        YhatVal = PhiVal*Theta;

        YhatMatrixVal = zeros(XdimVal,XdimVal);
        
        for i = 1:1:XdimVal
            for j = 1:1:XdimVal
                 YhatMatrixVal(i,j) = YhatVal((i-1)*XdimVal+j);
            end    
        end

        MseVal = sum((YflatVal-YhatVal').^2)/length(YflatVal);
        MseValV(CdimOp) = MseVal;
             
end

% Plots the MSE for the series of centers chosen at the previous step
% for both identification and validation

figure
plot(CdimFrom:1:CdimTo,MseV(CdimFrom:1:CdimTo));
hold on
plot(CdimFrom:1:CdimTo,MseValV(CdimFrom:1:CdimTo));
title('Mean Square Error'); xlabel('Dimension of the Center Matrix'); ylabel('Value of the Error')
axis([CdimFrom CdimTo 0 0.2]);
