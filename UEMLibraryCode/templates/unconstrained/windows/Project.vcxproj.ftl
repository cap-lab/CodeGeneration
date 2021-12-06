<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|Win32">
      <Configuration>Debug</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|Win32">
      <Configuration>Release</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  
  <PropertyGroup Label="Globals">
    <Keyword>Win32Proj</Keyword>
  </PropertyGroup>
  <PropertyGroup Condition="'$(WindowsTargetPlatformVersion)'==''">
    <!-- Latest Target Version property -->
    <LatestTargetPlatformVersion>$([Microsoft.Build.Utilities.ToolLocationHelper]::GetLatestSDKTargetPlatformVersion('Windows', '10.0'))</LatestTargetPlatformVersion>
    <WindowsTargetPlatformVersion Condition="'$(WindowsTargetPlatformVersion)' == ''">$(LatestTargetPlatformVersion)</WindowsTargetPlatformVersion>
    <TargetPlatformVersion>$(WindowsTargetPlatformVersion)</TargetPlatformVersion>
  </PropertyGroup>
  
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup>
    <ConfigurationType>Application</ConfigurationType>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'" Label="Configuration">
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>$(DefaultPlatformToolset)</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'" Label="Configuration">
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>$(DefaultPlatformToolset)</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>$(DefaultPlatformToolset)</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>$(DefaultPlatformToolset)</PlatformToolset>
  </PropertyGroup>
  
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings">
  </ImportGroup>
  <ImportGroup Label="Shared">
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
    <Import Project="ProjectProperty.props" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
    <Import Project="ProjectProperty.props" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
    <Import Project="ProjectProperty.props" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
    <Import Project="ProjectProperty.props" />
  </ImportGroup>
  
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup>
    <LinkIncremental>true</LinkIncremental>
    <IncludePath>$(VC_IncludePath);$(WindowsSDK_IncludePath);$(IncludePath);</IncludePath>
    <ExcludePath>$(VC_IncludePath);$(WindowsSDK_IncludePath);</ExcludePath>
    <TargetName>proc</TargetName>
  </PropertyGroup>
  
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <ClCompile>
      <PreprocessorDefinitions>WIN32;_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
      <Optimization>Disabled</Optimization>
    </ClCompile>
    <Link>
      <TargetMachine>MachineX86</TargetMachine>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <SubSystem>Console</SubSystem>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <ClCompile>
      <PreprocessorDefinitions>WIN32;NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
    </ClCompile>
    <Link>
      <TargetMachine>MachineX86</TargetMachine>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <LanguageStandard>
      </LanguageStandard>
      <CompileAs>Default</CompileAs>
      <RemoveUnreferencedCodeData>true</RemoveUnreferencedCodeData>
      <DisableLanguageExtensions>false</DisableLanguageExtensions>
      <ExceptionHandling>Sync</ExceptionHandling>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
    </Link>
  </ItemDefinitionGroup>

  <ItemGroup>
  	<#list build_info.mainSourceList as source_file>
		<ClCompile Include="$(MAIN_DIR)\${source_file}"/>
	</#list>
 	<#list build_info.taskSourceCodeList as source_file>
 		<ClCompile Include="$(APPLICATION_DIR)\${source_file}"/>
 	</#list>
  	<#if build_info.extraSourceCodeSet??>
  	<#list build_info.extraSourceCodeSet as source_file>
  		<ClCompile Include="$(APPLICATION_DIR)\${source_file}"/>
  	</#list>
  	</#if>
  	<#list build_info.apiSourceList as source_file>
  		<ClCompile Include="$(API_DIR)\${source_file}"/>
  	</#list>
  	<#list build_info.kernelDataSourceList as source_file>
  		<ClCompile Include="$(KERNEL_DIR)\$(GENERATED_DIR)\${source_file}"/>
  	</#list>
  	<#list build_info.kernelSourceList as source_file>
  		<ClCompile Include="$(KERNEL_DIR)\${source_file}"/>
  	</#list>	
  	<#list build_info.kernelDeviceSourceList as source_file>
  		<ClCompile Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\${source_file}"/>
  	</#list>
  	<#list build_info.commonSourceList as source_file>
  		<ClCompile Include="$(COMMON_DIR)\${source_file}"/>
  	</#list>
  	<#list build_info.moduleSourceList as source_file>
  		<ClCompile Include="$(MODULE_DIR)\${source_file}"/>
  	</#list>
  </ItemGroup>
  <ItemGroup>
    <Text Include="$(APPLICATION_DIR)\*.cic" />
	<Text Include="$(APPLICATION_DIR)\*.cicl" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="*.h"/>
	<ClInclude Include="$(MAIN_DIR)\$(DEVICE_RESTRICTION)\*.h"/>
	<ClInclude Include="$(API_DIR)\$(INCLUDE_DIR)\*.h"/>
	<ClInclude Include="$(COMMON_DIR)\$(INCLUDE_DIR)\*.h"/>
	<ClInclude Include="$(COMMON_DIR)\$(INCLUDE_DIR)\$(ENCRYPTION_DIR)\*.h"/>
	<ClInclude Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(INCLUDE_DIR)\*.h"/>
	<#if (build_info.usedPeripheralList?size > 0) >
	<#list build_info.usedPeripheralList as peripheralName>
	<ClInclude Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(INCLUDE_DIR)\${peripheralName}\*.h"/>
	</#list>
	</#if>	
	<ClInclude Include="$(KERNEL_DIR)\$(INCLUDE_DIR)\*.h"/>
	<ClInclude Include="$(KERNEL_DIR)\$(INCLUDE_DIR)\$(ENCRYPTION_DIR)\*.h"/>
	<ClInclude Include="$(KERNEL_DIR)\$(GENERATED_DIR)\*.h"/>
	<ClInclude Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\$(INCLUDE_DIR)\*.h"/>
	<#if (build_info.usedPeripheralList?size > 0) >
	<#list build_info.usedPeripheralList as peripheralName>
	<ClInclude Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\$(INCLUDE_DIR)\${peripheralName}\*.h"/>
	</#list>
	</#if>
	<ClInclude Include="$(MODULE_DIR)\$(INCLUDE_DIR)\*.h"/>
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets">
  </ImportGroup>	
  </Project>