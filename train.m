% addpath(fullfile(getenv('GUROBI_HOME'), 'matlab'));

I = 1 : 4; % input nodes
H = 8 : 17; % "candidate" hidden nodes
O = 5 : 7; % output nodes
V = sort([ I H O ]); % all nodes

T = 1 : 50; % time steps
D = 1 : 1; % possible delays

numNodes = numel(V);
numEdges = numNodes * (numNodes - 1) * numel(D);
numClasses = numel(O);
outNode = O; % outNode(c + 1) is output node for class c

% Create the "candidate" edges.

E = zeros(numEdges, 3);

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

% Read in the dataset.

spikes = readtable('iris/iris_encoded_training_data.csv');
y = readmatrix('iris/iris_training_label.csv');

numSamples = max(spikes.sample);

msg = 'number of samples must match number of labels';
assert(numSamples == numel(y), msg);

X = zeros(numSamples, numNodes, numel(T)); % input spike data

for i = 1 : height(spikes)
  n = spikes.sample(i); % training sample n
  v = spikes.node(i); % input node v
  t = spikes.time(i); % time t
  X(n, v, t) = 1; % there is an input spike
end

% These indices help us index into our decision vector.

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

indices.f = reshape( ...
  next + (1 : numSamples * numClasses * numel(T)), ...
  [numSamples, numClasses, numel(T)] ...
);

next = next + numSamples * numClasses * numel(T);

numVars = next; % gives the dimension of our decision vector

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
  numSamples * numClasses * ...
  (1 + 3 * (numel(T) - 1));

numConstraints = numConstraints + ...
  numSamples * (numClasses - 1) * numel(T);

numConstraints = numConstraints + ...
  sum(outNode(y + 1) ~= min(outNode));

% These help us build our constraint matrix later on

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

      model.rhs(row) = 0;
      model.sense(row) = '<';

      row = row + 1;

      % q(n, e, t) - s(n, u, t - d) <= 0

      rowIndices(index) = row;
      colIndices(index) = qCol;
      val(index) = 1;

      index = index + 1;

      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = -1;

      index = index + 1;

      model.rhs(row) = 0;
      model.sense(row) = '<';

      row = row + 1;

      % -q(n, e, t) + z(e) + s(n, u, t - d) <= 1

      rowIndices(index) = row;
      colIndices(index) = qCol;
      val(index) = -1;

      index = index + 1;

      rowIndices(index) = row;
      colIndices(index) = zCol;
      val(index) = 1;

      index = index + 1;

      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;

      index = index + 1;

      model.rhs(row) = 1;
      model.sense(row) = '<';

      row = row + 1;

    end

  end

end

% Add the arrival constraints.

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

        rowIndices(index) = row;
        colIndices(index) = qCol;
        val(index) = 1;

        index = index + 1;

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

      for qCol = qCols

        rowIndices(index) = row;
        colIndices(index) = qCol;
        val(index) = -1;

        index = index + 1;

      end      

      model.rhs(row) = 0;
      model.sense(row) = '<';

      row = row + 1;

    end

  end

end

% Add the firing constraints.

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

      model.rhs(row) = xVal;
      model.sense(row) = '>';

      row = row + 1;

      % s(n, v, t) >= b(n, v, t)

      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;

      index = index + 1;

      rowIndices(index) = row;
      colIndices(index) = bCol;
      val(index) = -1;

      index = index + 1;

      model.rhs(row) = 0;
      model.sense(row) = '>';

      row = row + 1;

      % s(n, v, t) <= X(n, v, t) + b(n, v, t)

      rowIndices(index) = row;
      colIndices(index) = sCol;
      val(index) = 1;

      index = index + 1;

      rowIndices(index) = row;
      colIndices(index) = bCol;
      val(index) = -1;

      index = index + 1;

      model.rhs(row) = xVal;
      model.sense(row) = '<';

      row = row + 1;

    end

  end

end

% Add the hidden neuron activation constraints.

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

    rowIndices(index) = row;
    colIndices(index) = gCol;
    val(index) = -1;

    index = index + 1;

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

    rowIndices(index) = row;
    colIndices(index) = gCol;
    val(index) = -1;

    index = index + 1;

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

  for zCol = zCols

    rowIndices(index) = row;
    colIndices(index) = zCol;
    val(index) = -1;

    index = index + 1;

  end

  model.rhs(row) = 0;
  model.sense(row) = '<';

  row = row + 1;

end

% Add the structural constraints.

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

      model.rhs(row) = 1;
      model.sense(row) = '<';

      row = row + 1;

    end

  end

end

% Add the cumulative firing constraints.

for n = 1 : numSamples

  for j = 1 : numClasses

    o = outNode(j);

    for t = 1 : numel(T)

      fCol = indices.f(n, j, t);
      sCol = indices.s(n, o, t);

      if t == 1

        % f(n, j, 1) = s(n, o, 1)

        rowIndices(index) = row;
        colIndices(index) = fCol;
        val(index) = 1;

        index = index + 1;

        rowIndices(index) = row;
        colIndices(index) = sCol;
        val(index) = -1;

        index = index + 1;

        model.rhs(row) = 0;
        model.sense(row) = '=';

        row = row + 1;

      else

        prevFCol = indices.f(n, j, t - 1);

        % f(n, j, t) >= s(n, o, t)

        rowIndices(index) = row;
        colIndices(index) = fCol;
        val(index) = -1;

        index = index + 1;

        rowIndices(index) = row;
        colIndices(index) = sCol;
        val(index) = 1;

        index = index + 1;

        model.rhs(row) = 0;
        model.sense(row) = '<';

        row = row + 1;

        % f(n, j, t) >= f(n, j, t - 1)

        rowIndices(index) = row;
        colIndices(index) = fCol;
        val(index) = -1;

        index = index + 1;

        rowIndices(index) = row;
        colIndices(index) = prevFCol;
        val(index) = 1;

        index = index + 1;

        model.rhs(row) = 0;
        model.sense(row) = '<';

        row = row + 1;

        % f(n, j, t) <= f(n, j, t - 1) + s(n, o, t)

        rowIndices(index) = row;
        colIndices(index) = fCol;
        val(index) = 1;

        index = index + 1;

        rowIndices(index) = row;
        colIndices(index) = prevFCol;
        val(index) = -1;

        index = index + 1;

        rowIndices(index) = row;
        colIndices(index) = sCol;
        val(index) = -1;

        index = index + 1;

        model.rhs(row) = 0;
        model.sense(row) = '<';

        row = row + 1;

      end

    end

  end

end

% Add the decoded label constraints.

for n = 1 : numSamples

  trueClass = y(n);
  trueOutput = outNode(trueClass + 1);
  trueIndex = find(outNode == trueOutput);
  hCol = indices.h(n);

  for j = 1 : numClasses

    otherOutput = outNode(j);

    if otherOutput == trueOutput
      continue
    end

    for t = 1 : numel(T)

      otherFCol = indices.f(n, j, t);

      if otherOutput > trueOutput
        trueFCol = indices.f(n, trueIndex, t);
      else
        if t == 1
          trueFCol = [];
        else
          trueFCol = indices.f(n, trueIndex, t - 1);
        end
      end

      % f(n, other, t) - f(n, true, t or t - 1) + h(n) <= 1

      rowIndices(index) = row;
      colIndices(index) = otherFCol;
      val(index) = 1;

      index = index + 1;

      if ~isempty(trueFCol)

        rowIndices(index) = row;
        colIndices(index) = trueFCol;
        val(index) = -1;

        index = index + 1;

      end

      rowIndices(index) = row;
      colIndices(index) = hCol;
      val(index) = 1;

      index = index + 1;

      model.rhs(row) = 1;
      model.sense(row) = '<';

      row = row + 1;

    end

  end

  if trueOutput ~= min(outNode)

    rowIndices(index) = row;
    colIndices(index) = hCol;
    val(index) = 1;

    index = index + 1;

    for j = 1 : numClasses

      fCol = indices.f(n, j, numel(T));

      rowIndices(index) = row;
      colIndices(index) = fCol;
      val(index) = -1;

      index = index + 1;

    end

    model.rhs(row) = 0;
    model.sense(row) = '<';

    row = row + 1;

  end

end

assert(row - 1 == numConstraints, 'bad constraint count');

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

params.MIPFocus = 1;
params.TimeLimit = 82800;
params.Threads = 48;
params.Seed = 1;
params.OutputFlag = 1;

result = gurobi(model, params);

if isfield(result, 'x')
  for n = 1 : numSamples
    fprintf('Sample %d:\n', n);
    for v = 1 : numNodes
      spikeIndices = reshape(indices.s(n, v, :), [], 1);
      spikeRaster = round(result.x(spikeIndices))';
      rasterString = char('0' + spikeRaster);
      fprintf('Node %2d: %s\n', v, rasterString);
    end
  end
end

tol = 0.5;

activeHiddenNodes = H(result.x(indices.g) > tol);

fprintf('Active hidden nodes: [');
if numel(activeHiddenNodes) ~= 0
  fprintf(' %d', activeHiddenNodes(1));
  for i = 2 : numel(activeHiddenNodes)
    fprintf(', %d', activeHiddenNodes(i));
  end
end
fprintf(' ]\n');

activeEdges = E(result.x(indices.z) > tol, :);

fprintf('Active edges: [');
if size(activeEdges, 1) ~= 0
  activeEdge = activeEdges(1, :);
  fprintf(' (%d, %d, %d)', activeEdge(1), activeEdge(2), activeEdge(3));
  for i = 2 : size(activeEdges, 1)
    activeEdge = activeEdges(i, :);
    fprintf(', (%d, %d, %d)', activeEdge(1), activeEdge(2), activeEdge(3));
  end
end
fprintf(' ]\n');
