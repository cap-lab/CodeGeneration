<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets" />
  <ImportGroup Label="PropertySheets">
    <Import Project="DirProperty.props" />
  </ImportGroup>
  <PropertyGroup>
    <IncludePath>$(VC_IncludePath);$(WindowsSDK_IncludePath);$(IncludePath);</IncludePath>
    <ExcludePath>$(VC_IncludePath);$(WindowsSDK_IncludePath);$(MSBuild_ExecutablePath);$(VC_LibraryPath_x86);$(ProjectDir)$(COMMON_DIR)\constrained;$(ProjectDir)$(COMMON_DIR)\unconstrained\native\linux;$(ProjectDir)$(KERNEL_DIR)\constrained;$(ProjectDir)$(MAIN_DIR)\constrained;$(ProjectDir)$(MAIN_DIR)\unconstrained\native\linux;$(ProjectDir)$(MODULE_DIR);$(ExcludePath)</ExcludePath>
  </PropertyGroup>
  <ItemDefinitionGroup>
    <ClCompile>
      <AdditionalIncludeDirectories>$(ProjectDir)$(MAIN_DIR)\include;$(ProjectDir)$(API_DIR)\include;$(ProjectDir)$(KERNEL_DIR)\include;$(ProjectDir)$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\include;$(ProjectDir);$(ProjectDir)$(COMMON_DIR)\include;$(ProjectDir)$(COMMON_DIR)\$(DEVICE_RESTRICTION)\include;$(ProjectDir)$(COMMON_DIR)\$(DEVICE_RESTRICTION)\include\communication;$(ProjectDir)$(MODULE_DIR)\include;$(ProjectDir)$(MAIN_DIR)\$(DEVICE_RESTRICTION);$(ProjectDir)$(MAIN_DIR)\$(DEVICE_RESTRICTION)\$(NATIVE_DIR);$(ProjectDir)$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\include\communication;$(ProjectDir)$(COMMON_DIR)</AdditionalIncludeDirectories>
      <PreprocessorDefinitions>HAVE_CONFIG_H</PreprocessorDefinitions>
      <AdditionalOptions>${build_info.cflags} %(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
	<Link>
	  <AdditionalDependencies>ws2_32.lib;%(AdditionalDependencies)</AdditionalDependencies>
	  <AdditionalOptions>${build_info.ldflags} %(AdditionalOptions)</AdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <BuildMacro Include="MAIN_DIR">
      <Value>$(MAIN_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="APPLICATION_DIR">
      <Value>$(APPLICATION_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="API_DIR">
      <Value>$(API_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="KERNEL_DIR">
      <Value>$(KERNEL_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="COMMON_DIR">
      <Value>$(COMMON_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="MODULE_DIR">
      <Value>$(MODULE_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="PLATFORM_DIR">
      <Value>$(PLATFORM_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="DEVICE_RESTRICTION">
      <Value>$(DEVICE_RESTRICTION)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
    <BuildMacro Include="NATIVE_DIR">
      <Value>$(NATIVE_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
	<BuildMacro Include="INCLUDE_DIR">
      <Value>$(INCLUDE_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
	<BuildMacro Include="GENERATED_DIR">
      <Value>$(GENERATED_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
	<BuildMacro Include="COMMUNICATION_DIR">
      <Value>$(COMMUNICATION_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
	<BuildMacro Include="TCP_DIR">
      <Value>$(TCP_DIR)</Value>
      <EnvironmentVariable>true</EnvironmentVariable>
    </BuildMacro>
  </ItemGroup>
</Project>