pathToDirectory = "C:\Users\itama\Desktop\Virmen_Green\05-Dec-2023 141012 Green_02_DavidParadigm";
A_B_lickport_fd = fopen(strcat(pathToDirectory, "\A-B_leakport_record.dat")); 
A_B_lickport_data = fread(A_B_lickport_fd,[5 inf], 'double');
A_B_lickport_data = A_B_lickport_data';
columnHeaders1 = {'timestamp', 'A_signal', 'B_signal', 'lickport_signal', 'trial_num'};  
A_B_lickport_Datatable = array2table(A_B_lickport_data, 'VariableNames', columnHeaders1);
writetable(A_B_lickport_Datatable, strcat(pathToDirectory, "\A-B_leakport_record.csv"));
fclose(A_B_lickport_fd);