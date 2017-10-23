within ;
model sin_model
  Modelica.Blocks.Math.Sin sin
    annotation (Placement(transformation(extent={{0,-4},{20,16}})));
  Modelica.Blocks.Interfaces.RealInput u
    annotation (Placement(transformation(extent={{-128,-50},{-88,-10}})));
  Modelica.Blocks.Math.Gain gain(k=a)
    annotation (Placement(transformation(extent={{30,-4},{50,16}})));
  Modelica.Blocks.Sources.Clock clock
    annotation (Placement(transformation(extent={{-60,-4},{-40,16}})));
  Modelica.Blocks.Math.Gain gain1(k=b/10000)
    annotation (Placement(transformation(extent={{-30,-4},{-10,16}})));
  Modelica.Blocks.Math.Add add
    annotation (Placement(transformation(extent={{64,-10},{84,10}})));
  Modelica.Blocks.Interfaces.RealOutput y
    annotation (Placement(transformation(extent={{96,-10},{116,10}})));
  parameter Real b=1 "Gain value multiplied with input signal";
  parameter Real a=1 "Gain value multiplied with input signal";
equation
  connect(sin.y, gain.u)
    annotation (Line(points={{21,6},{21,6},{28,6}}, color={0,0,127}));
  connect(clock.y, gain1.u)
    annotation (Line(points={{-39,6},{-39,6},{-32,6}}, color={0,0,127}));
  connect(gain1.y, sin.u)
    annotation (Line(points={{-9,6},{-9,6},{-2,6}}, color={0,0,127}));
  connect(gain.y, add.u1)
    annotation (Line(points={{51,6},{62,6}}, color={0,0,127}));
  connect(add.y, y)
    annotation (Line(points={{85,0},{85,0},{106,0}}, color={0,0,127}));
  connect(u, add.u2) annotation (Line(points={{-108,-30},{52,-30},{52,-6},{62,
          -6}}, color={0,0,127}));
  annotation (uses(Modelica(version="3.2.2")));
end sin_model;
