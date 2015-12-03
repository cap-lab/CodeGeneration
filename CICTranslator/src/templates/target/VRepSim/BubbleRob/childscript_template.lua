simSetThreadSwitchTiming(2)

-- Get some handles first:
##GET_HANDLE_MOTOR_SENSOR

-- Choose a port that is probably not used (try to always use a similar code):
simSetThreadAutomaticSwitch(false)
local portNb=simGetIntegerParameter(sim_intparam_server_port_next)
local portStart=simGetIntegerParameter(sim_intparam_server_port_start)
local portRange=simGetIntegerParameter(sim_intparam_server_port_range)
local newPortNb=portNb+1
if (newPortNb>=portStart+portRange) then
	newPortNb=portStart
end
simSetIntegerParameter(sim_intparam_server_port_next,newPortNb)
simSetThreadAutomaticSwitch(true)

-- Check what OS we are using:
platf=simGetIntegerParameter(sim_intparam_platform)
if (platf==0) then
	pluginFile='v_repExtRemoteApi.dll'
end
if (platf==1) then
	pluginFile='libv_repExtRemoteApi.dylib'
end
if (platf==2) then
	pluginFile='libv_repExtRemoteApi.so'
end

-- Check if the required remote Api plugin is there:
moduleName=0
moduleVersion=0
index=0
pluginNotFound=true
while moduleName do
	moduleName,moduleVersion=simGetModuleName(index)
	if (moduleName=='RemoteApi') then
		pluginNotFound=false
	end
	index=index+1
end

if (pluginNotFound) then
	-- Plugin was not found
	simDisplayDialog('Error',"Remote Api plugin was not found. ('"..pluginFile.."')&&nSimulation will not run properly",sim_dlgstyle_ok,true,nil,{0.8,0,0,0,0,0},{0.5,0,0,1,1,1})
else
	-- Ok, we found the plugin.
	-- We first start the remote Api server service (this requires the v_repExtRemoteApi plugin):
	--simExtRemoteApiStart(portNb) -- this server function will automatically close again at simulation end
	##REMOTE_API_START_W_PORT	

	-- Now we start the client application:
	##LAUNCH_CLIENT_APP
	
	if (result==-1) then
		-- The executable could not be launched!
		##DISPLAY_ERROR_MESSAGE
	end
end

-- This thread ends here. The bubbleRob will however still be controlled by
-- the client application via the remote Api mechanism!