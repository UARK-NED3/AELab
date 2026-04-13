function [AC_coeffs, voltage_sample_at, voltage_vec]  = importACData(folder, ACUnit)

    fileName = fullfile(folder, strcat(ACUnit, '.AC'));
    file = fopen(fileName, 'rb');

    if file == -1
        msg = sprintf('file %s could not be opened\n', fileName);
        error(msg);
        return;
    end

    num_AC_coeff_sets = 0;
    % get filesize and estimate entries
    fseek(file, 0, 'eof');
    filesize = ftell(file);
    initial_size = ceil(ftell(file)/393*1.2); % Take 20%
    frewind(file);
    % initalize variables for performance issues
    AC_coeffs = NaN(4, 24, initial_size);

    while (ftell(file) < filesize )
        num_AC_coeff_sets = num_AC_coeff_sets+1;
        t0 = fread(file, 1, 'float64');
        f_d = fread(file, 1, 'float64');
        dc = fread(file, 1, 'float64');
        if isnan(dc)
            dc = 0;
        end
        n = fread(file, 1, 'uint8');
        if n ~= 0
            AC_real = fread(file, n, 'float64', 8);
            % extend to 23 coefficients
            AC_real = [AC_real' zeros(1, max(0, 23-length(AC_real)))]';
            % jump back to the first AC_Imag coeff
            fseek(file, -n*16+8, 'cof');
            AC_imag = fread(file, n, 'float64', 8);
            % extend to 23 coefficients
            AC_imag = [AC_imag' zeros(1, max(0, 23-length(AC_imag)))]';
            if (ftell(file) < filesize)
                % undo last fread skip
                fseek(file, -8, 'cof');
            end

AC_coeffs_real = [dc, AC_real'];
AC_coeffs_imag = [0, AC_imag'];
AC_coeffs(:,:,num_AC_coeff_sets) = [
                % real part
                AC_coeffs_real(1:24); 
                % imaginary part
                AC_coeffs_imag(1:24);
                % frequencies
                [0, cumsum(ones(1,23) * f_d)];
                % keep track of t_0
                [t0, zeros(1, 23)]];
        end
    end

    fclose(file);

    AC_coeffs=AC_coeffs(:, :, 1:num_AC_coeff_sets);
    voltage_sample_at = @voltage_sample_at;
    voltage_vec = @voltage_vec;
end

function v = voltage_sample_at(t, AC_coeffs)
    % find coefficient set to use
    idx = max(find(AC_coeffs(4,1,:) < t));
    if isempty(idx)
        idx = size(AC_coeffs,3);
    end
    t_0 = AC_coeffs(4,1,idx);
    r = AC_coeffs(1,:,idx);
    i = AC_coeffs(2,:,idx);
    f = AC_coeffs(3,:,idx);
    v = sum(r'.*cos(2*pi*f'*(t-t_0)) - i'.*sin(2*pi*f'*(t-t_0)));
end

function v = voltage_vec_int(t, AC_coeffs)
    % find coefficient set to use
    idx = max(find(AC_coeffs(4,1,:) < min(t)));
    if isempty(idx)
        idx = size(AC_coeffs,3);
    end
    t_0 = AC_coeffs(4,1,idx);
    r = AC_coeffs(1,:,idx);
    i = AC_coeffs(2,:,idx);
    f = AC_coeffs(3,:,idx);
    v = sum(r'.*cos(2*pi*f'.*(t-t_0)) - i'.*sin(2*pi*f'.*(t-t_0)));
end

function v = voltage_vec(t, AC_coeffs)
    t0 = AC_coeffs(4,1,:);
    t0 = t0(:)';

    mapping = cell2mat(arrayfun(@(t0)find(t0 < t, 1)-1, t0, 'UniformOutput', false));
    idx_vec = unique(mapping);

    v = [];
    start_idx = 1;
    for i = idx_vec
        if (i > 0)
            v = [v voltage_vec_int(t(start_idx:i), AC_coeffs)];
        end
        start_idx = i+1;
    end
    if (start_idx < length(t))
        v = [v voltage_vec_int(t(start_idx:end), AC_coeffs)];
    end
end
