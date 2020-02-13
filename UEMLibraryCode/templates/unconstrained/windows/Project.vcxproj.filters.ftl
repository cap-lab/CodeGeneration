<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets">
  	<Import Project="DirProperty.props" />
  </ImportGroup>
  <ItemGroup>
    <Filter Include="$(API_DIR)">
    </Filter>
    <Filter Include="$(SRC_DIR)">
    </Filter>
    <Filter Include="$(APPLICATION_DIR)">
		<Extensions>cic;cicl;</Extensions>
    </Filter>
    <Filter Include="$(COMMON_DIR)">
    </Filter>
    <Filter Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)">
    </Filter>
    <Filter Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(NATIVE_DIR)">
    </Filter>
    <Filter Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(PLATFORM_DIR)">
    </Filter>
    <Filter Include="$(KERNEL_DIR)">
    </Filter>
    <Filter Include="$(KERNEL_DIR)\$(GENERATED_DIR)">
    </Filter>
    <Filter Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)">
    </Filter>
    <Filter Include="$(MAIN_DIR)\$(DEVICE_RESTRICTION)\$(NATIVE_DIR)">
    </Filter>
    <Filter Include="$(MAIN_DIR)\$(DEVICE_RESTRICTION)">
    </Filter>
    <Filter Include="$(MAIN_DIR)">
    </Filter>
    <Filter Include="$(MAIN_DIR)\$(DEVICE_RESTRICTION)\$(PLATFORM_DIR)">
    </Filter>
    <Filter Include="$(MODULE_DIR)">
    </Filter>
    <Filter Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(COMMUNICATION_DIR)">
    </Filter>
    <Filter Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(PLATFORM_DIR)\$(COMMUNICATION_DIR)">
    </Filter>
    <Filter Include="$(COMMON_DIR)\$(DEVICE_RESTRICTION)\$(PLATFORM_DIR)\$(COMMUNICATION_DIR)\$(TCP_DIR)">
    </Filter>
    <Filter Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\$(COMMUNICATION_DIR)">
    </Filter>
    <Filter Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\$(COMMUNICATION_DIR)\$(TCP_DIR)">
    </Filter>
	<Filter Include="Header Files">
		<Extensions>h;hh;hpp;hxx;hm;inl;inc;xsd</Extensions>
    </Filter>
  </ItemGroup>  
  <ItemGroup>
  	<#list build_info.mainSourceList as source_file>
  	<ClCompile Include="$(MAIN_DIR)\${source_file}">
		<Filter>$(MAIN_DIR)<#if (source_file?last_index_of("\\") >= 0)>\${source_file[0..<source_file?last_index_of("\\")]}</#if></Filter>
	</ClCompile>
	</#list>
	<#list build_info.taskSourceCodeList as source_file>
	<ClCompile Include="$(APPLICATION_DIR)\${source_file}">
		<Filter>$(APPLICATION_DIR)</Filter>
	</ClCompile>
	</#list>
	<#if build_info.extraSourceCodeSet??><#list build_info.extraSourceCodeSet as source_file>
	<ClCompile Include="$(APPLICATION_DIR)\${source_file}">
		<Filter>$(APPLICATION_DIR)</Filter>
	</ClCompile>
	</#list></#if>
  	<#list build_info.apiSourceList as source_file>
  	<ClCompile Include="$(API_DIR)\${source_file}">
		<Filter>$(API_DIR)</Filter>
	</ClCompile>
	</#list>
  	<#list build_info.kernelDataSourceList as source_file>
  	<ClCompile Include="$(KERNEL_DIR)\$(GENERATED_DIR)\${source_file}">
		<Filter>$(KERNEL_DIR)\$(GENERATED_DIR)</Filter>
	</ClCompile>
	</#list>
  	<#list build_info.kernelSourceList as source_file>
  	<ClCompile Include="$(KERNEL_DIR)\${source_file}">
		<Filter>$(KERNEL_DIR)</Filter>
	</ClCompile>
	</#list>	
  	<#list build_info.kernelDeviceSourceList as source_file>
  	<ClCompile Include="$(KERNEL_DIR)\$(DEVICE_RESTRICTION)\${source_file}">
		<Filter>$(KERNEL_DIR)\$(DEVICE_RESTRICTION)<#if (source_file?last_index_of("\\") >= 0)>\${source_file[0..<source_file?last_index_of("\\")]}</#if></Filter>
	</ClCompile>
	</#list>
  	<#list build_info.commonSourceList as source_file>
  	<ClCompile Include="$(COMMON_DIR)\${source_file}">
		<Filter>$(COMMON_DIR)<#if (source_file?last_index_of("\\") >= 0)>\${source_file[0..<source_file?last_index_of("\\")]}</#if></Filter>
	</ClCompile>
	</#list>
  	<#list build_info.moduleSourceList as source_file>
  	<ClCompile Include="$(MODULE_DIR)\${source_file}">
		<Filter>$(MODULE_DIR)</Filter>
	</ClCompile>
	</#list>	
  </ItemGroup>  
</Project>