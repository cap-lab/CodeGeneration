
package hopes.cic.xml;

import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlElementDecl;
import javax.xml.bind.annotation.XmlRegistry;
import javax.xml.namespace.QName;


/**
 * This object contains factory methods for each 
 * Java content interface and Java element interface 
 * generated in the hopes.cic.xml package. 
 * <p>An ObjectFactory allows you to programatically 
 * construct new instances of the Java representation 
 * for XML content. The Java representation of XML 
 * content can consist of schema derived interfaces 
 * and classes representing the binding of schema 
 * type definitions, element declarations and model 
 * groups.  Factory methods for each of these are 
 * provided in this class.
 * 
 */
@XmlRegistry
public class ObjectFactory {

    private final static QName _CICAlgorithm_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Algorithm");
    private final static QName _CICArchitecture_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Architecture");
    private final static QName _CICConfiguration_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Configuration");
    private final static QName _CICControl_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Control");
    private final static QName _CICDeviceIO_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CICDeviceIO");
    private final static QName _CICGPUSetup_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_GPUSetup");
    private final static QName _CICMapping_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Mapping");
    private final static QName _CICModule_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Module");
    private final static QName _CICProfile_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Profile");
    private final static QName _CICSchedule_QNAME = new QName("http://peace.snu.ac.kr/CICXMLSchema", "CIC_Schedule");

    /**
     * Create a new ObjectFactory that can be used to create new instances of schema derived classes for package: hopes.cic.xml
     * 
     */
    public ObjectFactory() {
    }

    /**
     * Create an instance of {@link TaskGroupType }
     * 
     */
    public TaskGroupType createTaskGroupType() {
        return new TaskGroupType();
    }

    /**
     * Create an instance of {@link CICAlgorithmType }
     * 
     */
    public CICAlgorithmType createCICAlgorithmType() {
        return new CICAlgorithmType();
    }

    /**
     * Create an instance of {@link CICArchitectureType }
     * 
     */
    public CICArchitectureType createCICArchitectureType() {
        return new CICArchitectureType();
    }

    /**
     * Create an instance of {@link CICConfigurationType }
     * 
     */
    public CICConfigurationType createCICConfigurationType() {
        return new CICConfigurationType();
    }

    /**
     * Create an instance of {@link CICControlType }
     * 
     */
    public CICControlType createCICControlType() {
        return new CICControlType();
    }

    /**
     * Create an instance of {@link CICDeviceIOType }
     * 
     */
    public CICDeviceIOType createCICDeviceIOType() {
        return new CICDeviceIOType();
    }

    /**
     * Create an instance of {@link CICGPUSetupType }
     * 
     */
    public CICGPUSetupType createCICGPUSetupType() {
        return new CICGPUSetupType();
    }

    /**
     * Create an instance of {@link CICMappingType }
     * 
     */
    public CICMappingType createCICMappingType() {
        return new CICMappingType();
    }

    /**
     * Create an instance of {@link CICModuleType }
     * 
     */
    public CICModuleType createCICModuleType() {
        return new CICModuleType();
    }

    /**
     * Create an instance of {@link CICProfileType }
     * 
     */
    public CICProfileType createCICProfileType() {
        return new CICProfileType();
    }

    /**
     * Create an instance of {@link CICScheduleType }
     * 
     */
    public CICScheduleType createCICScheduleType() {
        return new CICScheduleType();
    }

    /**
     * Create an instance of {@link VectorType }
     * 
     */
    public VectorType createVectorType() {
        return new VectorType();
    }

    /**
     * Create an instance of {@link VectorListType }
     * 
     */
    public VectorListType createVectorListType() {
        return new VectorListType();
    }

    /**
     * Create an instance of {@link TimeType }
     * 
     */
    public TimeType createTimeType() {
        return new TimeType();
    }

    /**
     * Create an instance of {@link SizeType }
     * 
     */
    public SizeType createSizeType() {
        return new SizeType();
    }

    /**
     * Create an instance of {@link TaskListType }
     * 
     */
    public TaskListType createTaskListType() {
        return new TaskListType();
    }

    /**
     * Create an instance of {@link LibraryListType }
     * 
     */
    public LibraryListType createLibraryListType() {
        return new LibraryListType();
    }

    /**
     * Create an instance of {@link ChannelListType }
     * 
     */
    public ChannelListType createChannelListType() {
        return new ChannelListType();
    }

    /**
     * Create an instance of {@link MulticastGroupListType }
     * 
     */
    public MulticastGroupListType createMulticastGroupListType() {
        return new MulticastGroupListType();
    }

    /**
     * Create an instance of {@link PortMapListType }
     * 
     */
    public PortMapListType createPortMapListType() {
        return new PortMapListType();
    }

    /**
     * Create an instance of {@link LibraryConnectionListType }
     * 
     */
    public LibraryConnectionListType createLibraryConnectionListType() {
        return new LibraryConnectionListType();
    }

    /**
     * Create an instance of {@link TaskGroupListType }
     * 
     */
    public TaskGroupListType createTaskGroupListType() {
        return new TaskGroupListType();
    }

    /**
     * Create an instance of {@link ModeListType }
     * 
     */
    public ModeListType createModeListType() {
        return new ModeListType();
    }

    /**
     * Create an instance of {@link TaskType }
     * 
     */
    public TaskType createTaskType() {
        return new TaskType();
    }

    /**
     * Create an instance of {@link ExternalTaskType }
     * 
     */
    public ExternalTaskType createExternalTaskType() {
        return new ExternalTaskType();
    }

    /**
     * Create an instance of {@link TaskDataParallelType }
     * 
     */
    public TaskDataParallelType createTaskDataParallelType() {
        return new TaskDataParallelType();
    }

    /**
     * Create an instance of {@link TaskPortType }
     * 
     */
    public TaskPortType createTaskPortType() {
        return new TaskPortType();
    }

    /**
     * Create an instance of {@link MulticastPortType }
     * 
     */
    public MulticastPortType createMulticastPortType() {
        return new MulticastPortType();
    }

    /**
     * Create an instance of {@link TaskModeType }
     * 
     */
    public TaskModeType createTaskModeType() {
        return new TaskModeType();
    }

    /**
     * Create an instance of {@link TaskRateType }
     * 
     */
    public TaskRateType createTaskRateType() {
        return new TaskRateType();
    }

    /**
     * Create an instance of {@link LibraryMasterPortType }
     * 
     */
    public LibraryMasterPortType createLibraryMasterPortType() {
        return new LibraryMasterPortType();
    }

    /**
     * Create an instance of {@link TaskParameterType }
     * 
     */
    public TaskParameterType createTaskParameterType() {
        return new TaskParameterType();
    }

    /**
     * Create an instance of {@link LibraryFunctionArgumentType }
     * 
     */
    public LibraryFunctionArgumentType createLibraryFunctionArgumentType() {
        return new LibraryFunctionArgumentType();
    }

    /**
     * Create an instance of {@link LibraryFunctionType }
     * 
     */
    public LibraryFunctionType createLibraryFunctionType() {
        return new LibraryFunctionType();
    }

    /**
     * Create an instance of {@link LibraryType }
     * 
     */
    public LibraryType createLibraryType() {
        return new LibraryType();
    }

    /**
     * Create an instance of {@link ChannelType }
     * 
     */
    public ChannelType createChannelType() {
        return new ChannelType();
    }

    /**
     * Create an instance of {@link ChannelPortType }
     * 
     */
    public ChannelPortType createChannelPortType() {
        return new ChannelPortType();
    }

    /**
     * Create an instance of {@link MulticastGroupType }
     * 
     */
    public MulticastGroupType createMulticastGroupType() {
        return new MulticastGroupType();
    }

    /**
     * Create an instance of {@link PortMapType }
     * 
     */
    public PortMapType createPortMapType() {
        return new PortMapType();
    }

    /**
     * Create an instance of {@link TaskLibraryConnectionType }
     * 
     */
    public TaskLibraryConnectionType createTaskLibraryConnectionType() {
        return new TaskLibraryConnectionType();
    }

    /**
     * Create an instance of {@link LibraryLibraryConnectionType }
     * 
     */
    public LibraryLibraryConnectionType createLibraryLibraryConnectionType() {
        return new LibraryLibraryConnectionType();
    }

    /**
     * Create an instance of {@link MTMType }
     * 
     */
    public MTMType createMTMType() {
        return new MTMType();
    }

    /**
     * Create an instance of {@link MTMModeListType }
     * 
     */
    public MTMModeListType createMTMModeListType() {
        return new MTMModeListType();
    }

    /**
     * Create an instance of {@link MTMModeType }
     * 
     */
    public MTMModeType createMTMModeType() {
        return new MTMModeType();
    }

    /**
     * Create an instance of {@link MTMVariableListType }
     * 
     */
    public MTMVariableListType createMTMVariableListType() {
        return new MTMVariableListType();
    }

    /**
     * Create an instance of {@link MTMVariableType }
     * 
     */
    public MTMVariableType createMTMVariableType() {
        return new MTMVariableType();
    }

    /**
     * Create an instance of {@link MTMTransitionListType }
     * 
     */
    public MTMTransitionListType createMTMTransitionListType() {
        return new MTMTransitionListType();
    }

    /**
     * Create an instance of {@link MTMTransitionType }
     * 
     */
    public MTMTransitionType createMTMTransitionType() {
        return new MTMTransitionType();
    }

    /**
     * Create an instance of {@link MTMConditionListType }
     * 
     */
    public MTMConditionListType createMTMConditionListType() {
        return new MTMConditionListType();
    }

    /**
     * Create an instance of {@link MTMConditionType }
     * 
     */
    public MTMConditionType createMTMConditionType() {
        return new MTMConditionType();
    }

    /**
     * Create an instance of {@link LoopStructureType }
     * 
     */
    public LoopStructureType createLoopStructureType() {
        return new LoopStructureType();
    }

    /**
     * Create an instance of {@link FaultToleranceType }
     * 
     */
    public FaultToleranceType createFaultToleranceType() {
        return new FaultToleranceType();
    }

    /**
     * Create an instance of {@link HardwareDependencyType }
     * 
     */
    public HardwareDependencyType createHardwareDependencyType() {
        return new HardwareDependencyType();
    }

    /**
     * Create an instance of {@link HardwarePlatformType }
     * 
     */
    public HardwarePlatformType createHardwarePlatformType() {
        return new HardwarePlatformType();
    }

    /**
     * Create an instance of {@link ModeType }
     * 
     */
    public ModeType createModeType() {
        return new ModeType();
    }

    /**
     * Create an instance of {@link ModeTaskType }
     * 
     */
    public ModeTaskType createModeTaskType() {
        return new ModeTaskType();
    }

    /**
     * Create an instance of {@link ModeTaskGroupType }
     * 
     */
    public ModeTaskGroupType createModeTaskGroupType() {
        return new ModeTaskGroupType();
    }

    /**
     * Create an instance of {@link HeaderListType }
     * 
     */
    public HeaderListType createHeaderListType() {
        return new HeaderListType();
    }

    /**
     * Create an instance of {@link ArchitectureElementTypeListType }
     * 
     */
    public ArchitectureElementTypeListType createArchitectureElementTypeListType() {
        return new ArchitectureElementTypeListType();
    }

    /**
     * Create an instance of {@link ArchitectureElementListType }
     * 
     */
    public ArchitectureElementListType createArchitectureElementListType() {
        return new ArchitectureElementListType();
    }

    /**
     * Create an instance of {@link ArchitectureConnectionListType }
     * 
     */
    public ArchitectureConnectionListType createArchitectureConnectionListType() {
        return new ArchitectureConnectionListType();
    }

    /**
     * Create an instance of {@link ArchitectureDeviceListType }
     * 
     */
    public ArchitectureDeviceListType createArchitectureDeviceListType() {
        return new ArchitectureDeviceListType();
    }

    /**
     * Create an instance of {@link ArchitectureDeviceType }
     * 
     */
    public ArchitectureDeviceType createArchitectureDeviceType() {
        return new ArchitectureDeviceType();
    }

    /**
     * Create an instance of {@link DeviceConnectionListType }
     * 
     */
    public DeviceConnectionListType createDeviceConnectionListType() {
        return new DeviceConnectionListType();
    }

    /**
     * Create an instance of {@link TCPConnectionType }
     * 
     */
    public TCPConnectionType createTCPConnectionType() {
        return new TCPConnectionType();
    }

    /**
     * Create an instance of {@link UDPConnectionType }
     * 
     */
    public UDPConnectionType createUDPConnectionType() {
        return new UDPConnectionType();
    }

    /**
     * Create an instance of {@link SerialConnectionType }
     * 
     */
    public SerialConnectionType createSerialConnectionType() {
        return new SerialConnectionType();
    }

    /**
     * Create an instance of {@link ModuleListType }
     * 
     */
    public ModuleListType createModuleListType() {
        return new ModuleListType();
    }

    /**
     * Create an instance of {@link ModuleType }
     * 
     */
    public ModuleType createModuleType() {
        return new ModuleType();
    }

    /**
     * Create an instance of {@link EnvironmentVariableListType }
     * 
     */
    public EnvironmentVariableListType createEnvironmentVariableListType() {
        return new EnvironmentVariableListType();
    }

    /**
     * Create an instance of {@link EnvironmentVariableType }
     * 
     */
    public EnvironmentVariableType createEnvironmentVariableType() {
        return new EnvironmentVariableType();
    }

    /**
     * Create an instance of {@link ArchitectureElementTypeType }
     * 
     */
    public ArchitectureElementTypeType createArchitectureElementTypeType() {
        return new ArchitectureElementTypeType();
    }

    /**
     * Create an instance of {@link ArchitectureElementSlavePortType }
     * 
     */
    public ArchitectureElementSlavePortType createArchitectureElementSlavePortType() {
        return new ArchitectureElementSlavePortType();
    }

    /**
     * Create an instance of {@link ArchitectureElementType }
     * 
     */
    public ArchitectureElementType createArchitectureElementType() {
        return new ArchitectureElementType();
    }

    /**
     * Create an instance of {@link ArchitectureConnectType }
     * 
     */
    public ArchitectureConnectType createArchitectureConnectType() {
        return new ArchitectureConnectType();
    }

    /**
     * Create an instance of {@link ArchitectureConnectionSlaveType }
     * 
     */
    public ArchitectureConnectionSlaveType createArchitectureConnectionSlaveType() {
        return new ArchitectureConnectionSlaveType();
    }

    /**
     * Create an instance of {@link CodeGenerationType }
     * 
     */
    public CodeGenerationType createCodeGenerationType() {
        return new CodeGenerationType();
    }

    /**
     * Create an instance of {@link SimulationType }
     * 
     */
    public SimulationType createSimulationType() {
        return new SimulationType();
    }

    /**
     * Create an instance of {@link ControlTaskListType }
     * 
     */
    public ControlTaskListType createControlTaskListType() {
        return new ControlTaskListType();
    }

    /**
     * Create an instance of {@link ExclusiveControlTasksListType }
     * 
     */
    public ExclusiveControlTasksListType createExclusiveControlTasksListType() {
        return new ExclusiveControlTasksListType();
    }

    /**
     * Create an instance of {@link ControlTaskType }
     * 
     */
    public ControlTaskType createControlTaskType() {
        return new ControlTaskType();
    }

    /**
     * Create an instance of {@link ExclusiveControlTasksType }
     * 
     */
    public ExclusiveControlTasksType createExclusiveControlTasksType() {
        return new ExclusiveControlTasksType();
    }

    /**
     * Create an instance of {@link SensorListType }
     * 
     */
    public SensorListType createSensorListType() {
        return new SensorListType();
    }

    /**
     * Create an instance of {@link ActuatorListType }
     * 
     */
    public ActuatorListType createActuatorListType() {
        return new ActuatorListType();
    }

    /**
     * Create an instance of {@link DisplayListType }
     * 
     */
    public DisplayListType createDisplayListType() {
        return new DisplayListType();
    }

    /**
     * Create an instance of {@link SensorType }
     * 
     */
    public SensorType createSensorType() {
        return new SensorType();
    }

    /**
     * Create an instance of {@link SensorParameterType }
     * 
     */
    public SensorParameterType createSensorParameterType() {
        return new SensorParameterType();
    }

    /**
     * Create an instance of {@link SensorValueType }
     * 
     */
    public SensorValueType createSensorValueType() {
        return new SensorValueType();
    }

    /**
     * Create an instance of {@link ActuatorType }
     * 
     */
    public ActuatorType createActuatorType() {
        return new ActuatorType();
    }

    /**
     * Create an instance of {@link ActuatorParameterType }
     * 
     */
    public ActuatorParameterType createActuatorParameterType() {
        return new ActuatorParameterType();
    }

    /**
     * Create an instance of {@link ActuatorValueType }
     * 
     */
    public ActuatorValueType createActuatorValueType() {
        return new ActuatorValueType();
    }

    /**
     * Create an instance of {@link DisplayType }
     * 
     */
    public DisplayType createDisplayType() {
        return new DisplayType();
    }

    /**
     * Create an instance of {@link GPUTaskListType }
     * 
     */
    public GPUTaskListType createGPUTaskListType() {
        return new GPUTaskListType();
    }

    /**
     * Create an instance of {@link GPUTaskType }
     * 
     */
    public GPUTaskType createGPUTaskType() {
        return new GPUTaskType();
    }

    /**
     * Create an instance of {@link WorkSizeType }
     * 
     */
    public WorkSizeType createWorkSizeType() {
        return new WorkSizeType();
    }

    /**
     * Create an instance of {@link MappingGPUDeviceType }
     * 
     */
    public MappingGPUDeviceType createMappingGPUDeviceType() {
        return new MappingGPUDeviceType();
    }

    /**
     * Create an instance of {@link MappingGPUProcessorIdType }
     * 
     */
    public MappingGPUProcessorIdType createMappingGPUProcessorIdType() {
        return new MappingGPUProcessorIdType();
    }

    /**
     * Create an instance of {@link MappingTaskType }
     * 
     */
    public MappingTaskType createMappingTaskType() {
        return new MappingTaskType();
    }

    /**
     * Create an instance of {@link MappingExternalTaskType }
     * 
     */
    public MappingExternalTaskType createMappingExternalTaskType() {
        return new MappingExternalTaskType();
    }

    /**
     * Create an instance of {@link MappingDeviceType }
     * 
     */
    public MappingDeviceType createMappingDeviceType() {
        return new MappingDeviceType();
    }

    /**
     * Create an instance of {@link MappingLibraryType }
     * 
     */
    public MappingLibraryType createMappingLibraryType() {
        return new MappingLibraryType();
    }

    /**
     * Create an instance of {@link LibraryAccessItemType }
     * 
     */
    public LibraryAccessItemType createLibraryAccessItemType() {
        return new LibraryAccessItemType();
    }

    /**
     * Create an instance of {@link MappingProcessorIdType }
     * 
     */
    public MappingProcessorIdType createMappingProcessorIdType() {
        return new MappingProcessorIdType();
    }

    /**
     * Create an instance of {@link MappingMulticastType }
     * 
     */
    public MappingMulticastType createMappingMulticastType() {
        return new MappingMulticastType();
    }

    /**
     * Create an instance of {@link MappingMulticastConnectionType }
     * 
     */
    public MappingMulticastConnectionType createMappingMulticastConnectionType() {
        return new MappingMulticastConnectionType();
    }

    /**
     * Create an instance of {@link MappingMulticastUDPType }
     * 
     */
    public MappingMulticastUDPType createMappingMulticastUDPType() {
        return new MappingMulticastUDPType();
    }

    /**
     * Create an instance of {@link SoftwareModuleType }
     * 
     */
    public SoftwareModuleType createSoftwareModuleType() {
        return new SoftwareModuleType();
    }

    /**
     * Create an instance of {@link FileSourceListType }
     * 
     */
    public FileSourceListType createFileSourceListType() {
        return new FileSourceListType();
    }

    /**
     * Create an instance of {@link FileSourceType }
     * 
     */
    public FileSourceType createFileSourceType() {
        return new FileSourceType();
    }

    /**
     * Create an instance of {@link ProfileTaskModeType }
     * 
     */
    public ProfileTaskModeType createProfileTaskModeType() {
        return new ProfileTaskModeType();
    }

    /**
     * Create an instance of {@link ProfileTaskType }
     * 
     */
    public ProfileTaskType createProfileTaskType() {
        return new ProfileTaskType();
    }

    /**
     * Create an instance of {@link ProfileType }
     * 
     */
    public ProfileType createProfileType() {
        return new ProfileType();
    }

    /**
     * Create an instance of {@link ProfileExecutionBoundType }
     * 
     */
    public ProfileExecutionBoundType createProfileExecutionBoundType() {
        return new ProfileExecutionBoundType();
    }

    /**
     * Create an instance of {@link ProfileCommType }
     * 
     */
    public ProfileCommType createProfileCommType() {
        return new ProfileCommType();
    }

    /**
     * Create an instance of {@link ProfileMigrationType }
     * 
     */
    public ProfileMigrationType createProfileMigrationType() {
        return new ProfileMigrationType();
    }

    /**
     * Create an instance of {@link ProfileLibraryFunctionType }
     * 
     */
    public ProfileLibraryFunctionType createProfileLibraryFunctionType() {
        return new ProfileLibraryFunctionType();
    }

    /**
     * Create an instance of {@link ProfileLibraryType }
     * 
     */
    public ProfileLibraryType createProfileLibraryType() {
        return new ProfileLibraryType();
    }

    /**
     * Create an instance of {@link TaskGroupsType }
     * 
     */
    public TaskGroupsType createTaskGroupsType() {
        return new TaskGroupsType();
    }

    /**
     * Create an instance of {@link TaskGroupForScheduleType }
     * 
     */
    public TaskGroupForScheduleType createTaskGroupForScheduleType() {
        return new TaskGroupForScheduleType();
    }

    /**
     * Create an instance of {@link ScheduleGroupType }
     * 
     */
    public ScheduleGroupType createScheduleGroupType() {
        return new ScheduleGroupType();
    }

    /**
     * Create an instance of {@link ScheduleElementType }
     * 
     */
    public ScheduleElementType createScheduleElementType() {
        return new ScheduleElementType();
    }

    /**
     * Create an instance of {@link LoopType }
     * 
     */
    public LoopType createLoopType() {
        return new LoopType();
    }

    /**
     * Create an instance of {@link TaskInstanceType }
     * 
     */
    public TaskInstanceType createTaskInstanceType() {
        return new TaskInstanceType();
    }

    /**
     * Create an instance of {@link TaskGroupType.TaskGroup }
     * 
     */
    public TaskGroupType.TaskGroup createTaskGroupTypeTaskGroup() {
        return new TaskGroupType.TaskGroup();
    }

    /**
     * Create an instance of {@link TaskGroupType.Task }
     * 
     */
    public TaskGroupType.Task createTaskGroupTypeTask() {
        return new TaskGroupType.Task();
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICAlgorithmType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICAlgorithmType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Algorithm")
    public JAXBElement<CICAlgorithmType> createCICAlgorithm(CICAlgorithmType value) {
        return new JAXBElement<CICAlgorithmType>(_CICAlgorithm_QNAME, CICAlgorithmType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICArchitectureType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICArchitectureType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Architecture")
    public JAXBElement<CICArchitectureType> createCICArchitecture(CICArchitectureType value) {
        return new JAXBElement<CICArchitectureType>(_CICArchitecture_QNAME, CICArchitectureType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICConfigurationType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICConfigurationType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Configuration")
    public JAXBElement<CICConfigurationType> createCICConfiguration(CICConfigurationType value) {
        return new JAXBElement<CICConfigurationType>(_CICConfiguration_QNAME, CICConfigurationType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICControlType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICControlType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Control")
    public JAXBElement<CICControlType> createCICControl(CICControlType value) {
        return new JAXBElement<CICControlType>(_CICControl_QNAME, CICControlType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICDeviceIOType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICDeviceIOType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CICDeviceIO")
    public JAXBElement<CICDeviceIOType> createCICDeviceIO(CICDeviceIOType value) {
        return new JAXBElement<CICDeviceIOType>(_CICDeviceIO_QNAME, CICDeviceIOType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICGPUSetupType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICGPUSetupType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_GPUSetup")
    public JAXBElement<CICGPUSetupType> createCICGPUSetup(CICGPUSetupType value) {
        return new JAXBElement<CICGPUSetupType>(_CICGPUSetup_QNAME, CICGPUSetupType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICMappingType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICMappingType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Mapping")
    public JAXBElement<CICMappingType> createCICMapping(CICMappingType value) {
        return new JAXBElement<CICMappingType>(_CICMapping_QNAME, CICMappingType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICModuleType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICModuleType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Module")
    public JAXBElement<CICModuleType> createCICModule(CICModuleType value) {
        return new JAXBElement<CICModuleType>(_CICModule_QNAME, CICModuleType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICProfileType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICProfileType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Profile")
    public JAXBElement<CICProfileType> createCICProfile(CICProfileType value) {
        return new JAXBElement<CICProfileType>(_CICProfile_QNAME, CICProfileType.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link CICScheduleType }{@code >}
     * 
     * @param value
     *     Java instance representing xml element's value.
     * @return
     *     the new instance of {@link JAXBElement }{@code <}{@link CICScheduleType }{@code >}
     */
    @XmlElementDecl(namespace = "http://peace.snu.ac.kr/CICXMLSchema", name = "CIC_Schedule")
    public JAXBElement<CICScheduleType> createCICSchedule(CICScheduleType value) {
        return new JAXBElement<CICScheduleType>(_CICSchedule_QNAME, CICScheduleType.class, null, value);
    }

}
