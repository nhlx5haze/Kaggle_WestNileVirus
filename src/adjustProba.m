function retval = adjustProba (test_probs, data)

%Modify prob based on date 
for i=1:size(data,1)
    if (data(i,7) >= 39)
        test_probs(i) = test_probs(i) * 0.2;
    elseif (data(i,2) == 2012 && data(i,7) == 26)
        test_probs(i) = test_probs(i) * 0.8;
    elseif (data(i,2) == 2012 && data(i,7) >= 28 && data(i,7) <= 33)
        test_probs(i) = test_probs(i) * 0.6;
    elseif (data(i,2) == 2014 && data(i,7) >= 35 && data(i,7) <= 36)
        test_probs(i) = test_probs(i) * 0.6;
    else
        test_probs(i) = test_probs(i) * 0.4;
    end
end

%Modify prob based on Trap
k = find(data(:,end) == 0);
test_probs(k) = test_probs(k)-0.02;

k = find(data(:,end) > 0);
test_probs(k) = test_probs(k)+0.02;

%Modify prob based on species
k = find(data(:,11));
test_probs(k) = test_probs(k)*0.8;

k = find(data(:,12));
test_probs(k) = test_probs(k)*1.1;

retval = test_probs;

endfunction
