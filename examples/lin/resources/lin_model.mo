within ;
model lin_model
  Modelica.Blocks.Interfaces.RealInput u2
    annotation (Placement(transformation(extent={{-128,-46},{-88,-6}})));
  Modelica.Blocks.Math.Gain gain(k=b)
    annotation (Placement(transformation(extent={{-56,-36},{-36,-16}})));
  Modelica.Blocks.Math.Gain gain1(k=a)
    annotation (Placement(transformation(extent={{-56,14},{-36,34}})));
  Modelica.Blocks.Math.Add add
    annotation (Placement(transformation(extent={{64,-10},{84,10}})));
  Modelica.Blocks.Interfaces.RealOutput y
    annotation (Placement(transformation(extent={{96,-10},{116,10}})));
  parameter Real b=1 "Gain value multiplied with input signal";
  parameter Real a=1 "Gain value multiplied with input signal";
  Modelica.Blocks.Interfaces.RealInput u1
    annotation (Placement(transformation(extent={{-128,4},{-88,44}})));
equation
  connect(add.y, y)
    annotation (Line(points={{85,0},{85,0},{106,0}}, color={0,0,127}));
  connect(u1, gain1.u)
    annotation (Line(points={{-108,24},{-58,24}}, color={0,0,127}));
  connect(u2, gain.u) annotation (Line(points={{-108,-26},{-84,-26},{-58,-26}},
        color={0,0,127}));
  connect(gain1.y, add.u1) annotation (Line(points={{-35,24},{14,24},{14,6},{62,
          6}}, color={0,0,127}));
  connect(gain.y, add.u2) annotation (Line(points={{-35,-26},{14,-26},{14,-6},{
          62,-6}}, color={0,0,127}));
  annotation (uses(Modelica(version="3.2.2")));
end lin_model;
