package org.snu.cse.cap.translator.structure.device;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;

enum ArchitectureType {
	X86("x86"),
	X86_64("x86_64"),
	ARM("arm"),
	ARM64("arm64"),
	GENERIC("generic"),
	;
	
	private final String value;
	
	private ArchitectureType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static ArchitectureType fromValue(String value) {
		 for (ArchitectureType c : ArchitectureType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

enum SoftwarePlatformType {
	ARDUINO("arduino"),
	WINDOWS("windows"),
	LINUX("linux"),
	UCOS3("ucos-3"),
	;
	
	private final String value;
	
	private SoftwarePlatformType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static SoftwarePlatformType fromValue(String value) {
		 for (SoftwarePlatformType c : SoftwarePlatformType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

enum RuntimeType {
	NATIVE("native"),
	SOPHY("sophy"),
	HSIM("hsim"),
	;

	private final String value;
	
	private RuntimeType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static RuntimeType fromValue(String value) {
		 for (RuntimeType c : RuntimeType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

public class Device {
	private String name;
	private ArrayList<Processor> processorList;
	private HashMap<String, Connection> connectionList;
	private ArchitectureType architecture;
	private SoftwarePlatformType platform;
	private RuntimeType runtime;

	
	public Device(String name, String architecture, String platform, String runtime) 
	{
		this.name = name;
		this.architecture = ArchitectureType.fromValue(architecture);
		this.platform = SoftwarePlatformType.fromValue(platform);
		this.runtime = RuntimeType.fromValue(runtime);
		this.processorList = new ArrayList<Processor>();
		this.connectionList = new HashMap<String, Connection>();
	}
	
	public String getName() {
		return name;
	}

	public void putProcessingElement(int id, String name, ProcessorCategory type, int poolSize) 
	{
		Processor processor = new Processor(id, name, type, poolSize);
			
		this.processorList.add(processor);
	}
	
	public void putConnection(Connection connection) 
	{
		this.connectionList.put(connection.getName(), connection);
	}
	
	public Connection getConnection(String connectionName) throws InvalidDeviceConnectionException 
	{
		Connection connection;
		
		if(this.connectionList.containsKey(connectionName))
		{
			connection = this.connectionList.get(connectionName);	
		}
		else
		{
			throw new InvalidDeviceConnectionException();
		}
		
		return connection;
	}

	public ArchitectureType getArchitecture() {
		return architecture;
	}
	
	public SoftwarePlatformType getPlatform() {
		return platform;
	}
	
	public RuntimeType getRuntime() {
		return runtime;
	}
	
	public void setArchitecture(ArchitectureType architecture) {
		this.architecture = architecture;
	}
	
	public void setPlatform(SoftwarePlatformType platform) {
		this.platform = platform;
	}
	
	public void setRuntime(RuntimeType runtime) {
		this.runtime = runtime;
	}

	public ArrayList<Processor> getProcessorList() {
		return processorList;
	}

	public void setProcessorList(ArrayList<Processor> processorList) {
		this.processorList = processorList;
	}
}
