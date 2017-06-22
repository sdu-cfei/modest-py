within ;
model Simple2R1C
  parameter Modelica.SIunits.ThermalResistance R1=1
    "Constant thermal resistance of material";
  parameter Modelica.SIunits.ThermalResistance R2=1
    "Constant thermal resistance of material";
  parameter Modelica.SIunits.HeatCapacity C=1000
    "Heat capacity of element (= cp*m)";
  parameter Modelica.SIunits.Temperature Tstart=20
    "Initial temperature";


  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=C, T(fixed=
          true, start=Tstart))
    annotation (Placement(transformation(extent={{2,0},{22,20}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor thermalResistor2(R=
       R2) annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));
  Modelica.Blocks.Interfaces.RealInput Ti2
    annotation (Placement(transformation(extent={{-130,-20},{-90,20}})));
  Modelica.Blocks.Interfaces.RealOutput T
    annotation (Placement(transformation(extent={{96,-10},{116,10}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor thermalResistor1(R=
       R1) annotation (Placement(transformation(extent={{-42,36},{-22,56}})));
  Modelica.Blocks.Interfaces.RealInput Ti1
    annotation (Placement(transformation(extent={{-130,26},{-90,66}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor
    annotation (Placement(transformation(extent={{42,-10},{62,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
    prescribedTemperature1
    annotation (Placement(transformation(extent={{-78,36},{-58,56}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
    prescribedTemperature2
    annotation (Placement(transformation(extent={{-76,-10},{-56,10}})));
equation
  connect(heatCapacitor.port, thermalResistor2.port_b)
    annotation (Line(points={{12,0},{-20,0}}, color={191,0,0}));
  connect(thermalResistor1.port_b, heatCapacitor.port) annotation (Line(
        points={{-22,46},{-8,46},{-8,0},{12,0}}, color={191,0,0}));
  connect(heatCapacitor.port, temperatureSensor.port)
    annotation (Line(points={{12,0},{12,0},{42,0}}, color={191,0,0}));
  connect(T, temperatureSensor.T)
    annotation (Line(points={{106,0},{62,0}}, color={0,0,127}));
  connect(Ti1, prescribedTemperature1.T)
    annotation (Line(points={{-110,46},{-96,46},{-80,46}}, color={0,0,127}));
  connect(thermalResistor1.port_a, prescribedTemperature1.port)
    annotation (Line(points={{-42,46},{-50,46},{-58,46}}, color={191,0,0}));
  connect(Ti2, prescribedTemperature2.T)
    annotation (Line(points={{-110,0},{-94,0},{-78,0}}, color={0,0,127}));
  connect(thermalResistor2.port_a, prescribedTemperature2.port)
    annotation (Line(points={{-40,0},{-48,0},{-56,0}}, color={191,0,0}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="3.2.2")));
end Simple2R1C;
