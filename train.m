I = 1 : 4; % input neurons
H = 5 : 9; % "candidate" hidden neurons
O = 10 : 12; % output neurons
V = [ I H O ]; % all neurons

T = 1 : 50; % time steps
D = 1 : 10; % range of possible delay values

numNodes = numel(V);
numEdges = numNodes * (numNodes - 1) * numel(D);
numClasses = numel(O);
outNeuron = O; % outNeuron(c + 1) is output neuron for class c

E = zeros(numEdges, 3); % synapses

e = 1;
for u = V
  for v = V
    if u ~= v
      for d = D
        E(e, :) = [ u, v, d ];
        e = e + 1;
      end
    end
  end
end

spikes = readtable('encoded_iris_data.csv');
y = readmatrix('iris_label.csv');
y = y(:); % ensure y is a column vector

numSamples = max(spikes.sample);

msg = 'number of samples must match number of labels';
assert(numSamples == numel(y), msg);

X = zeros(numSamples, numNodes, numel(T)); % input spike data

for i = 1 : height(spikes)
  n = spikes.sample(i); % training sample n
  v = spikes.node(i); % input neuron v
  t = spikes.time(i); % time t
  X(n, v, t) = 1; % there is an input spike
end

indices = struct();

next = 0;

indices.z = next + (1 : numEdges);
next = next + numEdges;
indices.g = next + (1 : numel(H));
next = next + numel(H);
indices.s = reshape( ...
  next + (1 : numSamples * numNodes * numel(T)), ...
  [numSamples, numNodes, numel(T)] ...
);
next = next + numSamples * numNodes * numel(T);
indices.q = reshape( ...
  next + (1 : numSamples * numEdges * numel(T)), ...
  [numSamples, numEdges, numel(T)] ...
);
next = next + numSamples * numEdges * numel(T);
indices.b = reshape( ...
  next + (1 : numSamples * numNodes * numel(T)), ...
  [numSamples, numNodes, numel(T)] ...
);
next = next + numSamples * numNodes * numel(T);
indices.h = next + (1 : numSamples);
next = next + numSamples;

numVars = next; % gives the dimension of our decision vector x

% Calculate the number of constraints.

numConstraints = 0;

for n = 1 : numSamples
  for d = D
    numConstraints = numConstraints + ...
      (numNodes * (numNodes - 1)) * ...
      (d + 3 * (numel(T) - d));
  end
end

numConstraints = numConstraints + ...
  numSamples * numNodes * numel(T) * ...
  ((numNodes - 1) * numel(D) + 1);

numConstraints = numConstraints + ...
  numSamples * numNodes * numel(T) * 3;

numConstraints = numConstraints + ...
  numel(H) * (2 * (numNodes - 1) * numel(D) + 1);

numConstraints = numConstraints + ...
  numNodes * (numNodes - 1);

numConstraints = numConstraints + ...
  numSamples * (numClasses - 1);

% fprintf('numConstraints: %d\n', numConstraints);

rowIndices = zeros(4 * numConstraints, 1);
colIndices = zeros(4 * numConstraints, 1);
val = zeros(4 * numConstraints, 1);
index = 1;

model.rhs = zeros(numConstraints, 1);
model.sense = repmat('$', numConstraints, 1);
model.obj = zeros(numVars, 1);
model.lb = zeros(numVars, 1);
model.ub = ones(numVars, 1);
model.vtype = repmat('B', numVars, 1);

% Add the propagation constraints.

fprintf('%s: Adding the propagation constraints\n', datestr(now));

row = 1;

for n = 1 : numSamples
  for e = 1 : numEdges
    u = E(e, 1);
    d = E(e, 3);
    for t = 1 : min(d, numel(T))
      qCol = indices.q(n, e, t);
      rowIndices(index) = row;
      colIndices(index) = qCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, qCol) = 1;
      model.rhs(row) = 0;
      model.sense(row) = '=';
      row = row + 1;
    end
    for t = (d + 1) : numel(T)
      qCol = indices.q(n, e, t);
      zCol = indices.z(e);
      sCol = indices.s(n, u, t - d);

      % q(n, e, t) - z(e) <= 0
      rowIndices(index) = row;
      colIndices(index) = qCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, qCol) = 1;
      rowIndices(index) = row;
      colIndices(index) = zCol;
      val(index) = -1;
      index = index + 1;
      % model.A(row, zCol) = -1;
      model.rhs(row) = 0;
      model.sense(row) = '<';
      row = row + 1;

      % q(n, e, t) - s(n, u, t - d) <= 0
      rowIndices(index) = row;
      colIndices(index) = qCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, qCol) = 1;
      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = -1;
      index = index + 1;
      % model.A(row, sCol) = -1;
      model.rhs(row) = 0;
      model.sense(row) = '<';
      row = row + 1;

      % -q(n, e, t) + z(e) + s(n, u, t - d) <= 1
      rowIndices(index) = row;
      colIndices(index) = qCol;
      val(index) = -1;
      index = index + 1;
      % model.A(row, qCol) = -1;
      rowIndices(index) = row;
      colIndices(index) = zCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, zCol) = 1;
      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, sCol) = 1;
      model.rhs(row) = 1;
      model.sense(row) = '<';
      row = row + 1;
    end
  end
end

% Add the arrival constraints.

fprintf('%s: Adding the arrival constraints\n', datestr(now));

for n = 1 : numSamples
  for v = V
    for t = 1 : numel(T)
      bCol = indices.b(n, v, t);
      incomingEdges = find(E(:, 2) == v)';

      % b(n, v, t) >= q(n, e, t) for every edge e to v
      for e = incomingEdges
        qCol = indices.q(n, e, t);
        rowIndices(index) = row;
        colIndices(index) = bCol;
        val(index) = -1;
        index = index + 1;
        % model.A(row, bCol) = -1;
        rowIndices(index) = row;
        colIndices(index) = qCol;
        val(index) = 1;
        index = index + 1;
        % model.A(row, qCol) = 1;
        model.rhs(row) = 0;
        model.sense(row) = '<';
        row = row + 1;
      end

      % b(n, v, t) <= sum of q(n, e, t) for edges e to v
      qCols = indices.q(n, incomingEdges, t);
      rowIndices(index) = row;
      colIndices(index) = bCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, bCol) = 1;
      for qCol = qCols
        rowIndices(index) = row;
        colIndices(index) = qCol;
        val(index) = -1;
        index = index + 1;
      end      
      % model.A(row, qCols) = -1;
      model.rhs(row) = 0;
      model.sense(row) = '<';
      row = row + 1;
    end
  end
end

% Add the firing constraints.

fprintf('%s: Adding the firing constraints\n', datestr(now));

for n = 1 : numSamples
  for v = V
    for t = 1 : numel(T)
      sCol = indices.s(n, v, t);
      bCol = indices.b(n, v, t);
      xVal = X(n, v, t);

      % s(n, v, t) >= X(n, v, t)
      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, sCol) = 1;
      model.rhs(row) = xVal;
      model.sense(row) = '>';
      row = row + 1;

      % s(n, v, t) >= b(n, v, t)
      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, sCol) = 1;
      rowIndices(index) = row;
      colIndices(index) = bCol;
      val(index) = -1;
      index = index + 1;
      % model.A(row, bCol) = -1;
      model.rhs(row) = 0;
      model.sense(row) = '>';
      row = row + 1;

      % s(n, v, t) <= X(n, v, t) + b(n, v, t)
      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;
      index = index + 1;
      % model.A(row, sCol) = 1;
      rowIndices(index) = row;
      colIndices(index) = bCol;
      val(index) = -1;
      index = index + 1;
      % model.A(row, bCol) = -1;
      model.rhs(row) = xVal;
      model.sense(row) = '<';
      row = row + 1;
    end
  end
end

% Add the hidden neuron activation constraints.

fprintf('%s: Adding the hidden neuron constraints\n', datestr(now));

for h = H
  gCol = indices.g(h - H(1) + 1);
  incomingEdges = find(E(:, 2) == h)';
  outgoingEdges = find(E(:, 1) == h)';

  % z(e) <= g(h) for all edges e to h
  for e = incomingEdges
    zCol = indices.z(e);
    rowIndices(index) = row;
    colIndices(index) = zCol;
    val(index) = 1;
    index = index + 1;
    % model.A(row, zCol) = 1;
    rowIndices(index) = row;
    colIndices(index) = gCol;
    val(index) = -1;
    index = index + 1;
    % model.A(row, gCol) = -1;
    model.rhs(row) = 0;
    model.sense(row) = '<';
    row = row + 1;
  end

  % z(e) <= g(h) for all edges e from h
  for e = outgoingEdges
    zCol = indices.z(e);
    rowIndices(index) = row;
    colIndices(index) = zCol;
    val(index) = 1;
    index = index + 1;
    % model.A(row, zCol) = 1;
    rowIndices(index) = row;
    colIndices(index) = gCol;
    val(index) = -1;
    index = index + 1;
    % model.A(row, gCol) = -1;
    model.rhs(row) = 0;
    model.sense(row) = '<';
    row = row + 1;
  end

  % g(h) <= sum of all edges incident on h
  incidentEdges = [ incomingEdges outgoingEdges ];
  zCols = indices.z(incidentEdges);
  rowIndices(index) = row;
  colIndices(index) = gCol;
  val(index) = 1;
  index = index + 1;
  % model.A(row, gCol) = 1;
  for zCol = zCols
    rowIndices(index) = row;
    colIndices(index) = zCol;
    val(index) = -1;
    index = index + 1;
  end
  % model.A(row, zCols) = -1;
  model.rhs(row) = 0;
  model.sense(row) = '<';
  row = row + 1;
end

% Add the structural constraints.

fprintf('%s: Adding the structural constraints\n', datestr(now));

for u = V
  for v = V
    if u ~= v
      edgeRows = find(E(:, 1) == u & E(:, 2) == v)';
      zCols = indices.z(edgeRows);
      for zCol = zCols
        rowIndices(index) = row;
        colIndices(index) = zCol;
        val(index) = 1;
        index = index + 1;
      end
      % model.A(row, zCols) = 1;
      model.rhs(row) = 1;
      model.sense(row) = '<';
      row = row + 1;
    end
  end
end

% Add the decoded label constraints.

fprintf('%s: Adding the decoded label constraints\n', datestr(now));

M = numel(T) + 1;

for n = 1 : numSamples
  trueClass = y(n);
  hCol = indices.h(n);
  trueOutput = outNeuron(trueClass + 1);
  trueSCols = reshape( ...
    indices.s(n, trueOutput, :), 1, [] ...
  );
  for c = 0 : (numClasses - 1)
    if c == trueClass
      continue
    end
    otherOutput = outNeuron(c + 1);
    otherSCols = reshape( ...
      indices.s(n, otherOutput, :), 1, [] ...
    );

    % r(n, c) - r(n, trueClass) + M * h(n) <= rhs
    for otherSCol = otherSCols
      rowIndices(index) = row;
      colIndices(index) = otherSCol;
      val(index) = 1;
      index = index + 1;
    end
    % model.A(row, otherSCols) = 1;
    for trueSCol = trueSCols
      rowIndices(index) = row;
      colIndices(index) = trueSCol;
      val(index) = -1;
      index = index + 1;
    end
    % model.A(row, trueSCols) = -1;
    rowIndices(index) = row;
    colIndices(index) = hCol;
    val(index) = M;
    index = index + 1;
    % model.A(row, hCol) = M;
    if c < trueClass
      model.rhs(row) = M - 1;
    else
      model.rhs(row) = M;
    end
    model.sense(row) = '<';
    row = row + 1;
  end
end

assert(row - 1 == numConstraints, 'bad constraint count');

fprintf('%s: Creating the constraint matrix A\n', datestr(now));

model.A = sparse( ...
  rowIndices(1 : index - 1), ...
  colIndices(1 : index - 1), ...
  val(1 : index - 1), ...
  numConstraints, ...
  numVars ...
);

model.obj(:) = 0;
model.obj(indices.h) = -1;
model.modelsense = 'min';

result = gurobi(model);
