local _M = {}

local function _StrIsEmpty(s)
  return s == nil or s == ''
end

local function _LoadFollowee(data)
  local followee_list = {}
  for _, followee in ipairs(data) do
    local new_followee = {}
    new_followee["followee_id"] = tostring(followee)
    table.insert(followee_list, new_followee)
  end
  return followee_list
end


function _M.GetFollowee()
  local bridge_tracer = require "opentracing_bridge_tracer"
  local ngx = ngx
  local GenericObjectPool = require "GenericObjectPool"
  local SocialGraphServiceClient = require "social_network_SocialGraphService"
  local cjson = require "cjson"

  local req_id = tonumber(string.sub(ngx.var.request_id, 0, 15), 16)
  local tracer = bridge_tracer.new_from_global()
  local parent_span_context = tracer:binary_extract(
      ngx.var.opentracing_binary_context)
  local span = tracer:start_span("GetFollowee",
      {["references"] = {{"child_of", parent_span_context}}})
  local carrier = {}
  tracer:text_map_inject(span:context(), carrier)
 
  ngx.req.read_body()
  local args = ngx.req.get_uri_args()

  if (_StrIsEmpty(args.user_id)) then
    ngx.status = ngx.HTTP_BAD_REQUEST
    ngx.say("Incomplete arguments")
    ngx.log(ngx.ERR, "Incomplete arguments")
    span:finish()
    ngx.exit(ngx.HTTP_BAD_REQUEST)
  end

  local client = GenericObjectPool:connection(
    SocialGraphServiceClient, "social-graph-service.social-network.svc.cluster.local", 9090)
  local status, ret = pcall(client.GetFollowees, client, req_id,
      args.user_id, carrier)
  GenericObjectPool:returnConnection(client)
  if not status then
    ngx.status = ngx.HTTP_INTERNAL_SERVER_ERROR
    if (ret.message) then
      ngx.header.content_type = "text/plain"
      ngx.say("Get user-timeline failure: " .. ret.message)
      ngx.log(ngx.ERR, "Get user-timeline failure: " .. ret.message)
    else
      ngx.header.content_type = "text/plain"
      ngx.say("Get user-timeline failure: " .. ret.message)
      ngx.log(ngx.ERR, "Get user-timeline failure: " .. ret.message)
    end
    span:finish()
    ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
  else
    local followee_list = _LoadFollowee(ret)
    ngx.header.content_type = "application/json; charset=utf-8"
    ngx.say(cjson.encode(followee_list) )
  end

  span:finish()
end
return _M