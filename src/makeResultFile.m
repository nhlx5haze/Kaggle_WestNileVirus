function  makeResultFile( res, filename )

fileID = fopen(filename, 'w');
fprintf(fileID,'Id,WnvPresent\n');
cpt = 1;
for i=1:length(res)    
        fprintf(fileID,'%d,%f\n', cpt, res(i));    
        cpt = cpt + 1;
end

fclose(fileID);

end

